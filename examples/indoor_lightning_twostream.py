import os
import io
import argparse
import glob
import importlib

import numpy as np
from urllib.request import urlretrieve

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.plugins import DDPPlugin
from examples.str2bool import str2bool
import wandb
import MinkowskiEngine as ME

def plot_class_metric(trainer, pl_module, metric, plot_title):
    labels = trainer.datamodule.class_labels
    data = [[label, val] for (label, val) in zip(labels, metric)]
    for logger in pl_module.logger:
        if isinstance(logger, TensorBoardLogger):
            pass
        elif isinstance(logger, WandbLogger):
            table = wandb.Table(data=data, columns = ["label", plot_title.split('_')[-1]])
            logger.experiment.log({plot_title : wandb.plot.bar(table, "label",
                               plot_title.split('_')[-1], title=plot_title)})

class IoUPlotCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        plot_class_metric(trainer, pl_module, pl_module.train_class_iou.compute(), "train_ious")
        plot_class_metric(trainer, pl_module, pl_module.train_class_acc.compute(), "train_accs")
        pl_module.train_class_iou.reset()
        pl_module.train_class_acc.reset()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        plot_class_metric(trainer, pl_module, pl_module.val_class_iou.compute(), "val_ious")
        plot_class_metric(trainer, pl_module, pl_module.val_class_acc.compute(), "val_accs")
        pl_module.val_class_iou.reset()
        pl_module.val_class_acc.reset()


def plot_confusion_matrix(trainer, pl_module, confusion_metric, plot_title):
    conf_mat = confusion_metric.compute().detach().cpu().numpy()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                             display_labels=trainer.datamodule.class_labels)
    disp.plot(include_values=False, cmap="Blues", xticks_rotation='vertical')
    plt.title(plot_title)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    for logger in pl_module.logger:
        if isinstance(logger, TensorBoardLogger):
            # print('tb_log')
            logger.experiment.add_image(plot_title, im, global_step=trainer.current_epoch)
        # elif isinstance(logger, WandbLogger):
        #     # print('wandb_log')
        #     logger.experiment.log({plot_title: [wandb.Image(im)]})

    # print(pl_module.current_epoch)

    plt.close()

class ConfusionMatrixPlotCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        plot_confusion_matrix(trainer, pl_module, pl_module.train_conf_metrics["ConfusionMatrix"], "train_confusion_matrix")
        plot_confusion_matrix(trainer, pl_module, pl_module.train_conf_metrics["NormalizedConfusionMatrix"], "train_normalized_confusion_matrix")
        pl_module.train_conf_metrics.reset()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        plot_confusion_matrix(trainer, pl_module, pl_module.val_conf_metrics["ConfusionMatrix"], "val_confusion_matrix")
        plot_confusion_matrix(trainer, pl_module, pl_module.val_conf_metrics["NormalizedConfusionMatrix"], "val_normalized_confusion_matrix")
        pl_module.val_conf_metrics.reset()

def init_module_from_args(module, args=None, **kwargs):
    if module is None:
        return None
    parser = argparse.ArgumentParser()
    parser = module.add_argparse_args(parser)
    module_args, args = parser.parse_known_args(args=args)
    args_dict = vars(module_args)
    args_dict.update(kwargs)
    return module(**args_dict), args, args_dict

class MainArgs():
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MinkSegModel")
        parser.add_argument("--exp_name", type=str, default='default')
        parser.add_argument('--run_mode', type=str, default='train', choices=['train','validate','test', 'visualize'])
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument("--pl_module", type=str, default='examples.TwoStreamSeg.TwoStreamSegmentationModule')
        parser.add_argument("--pl_datamodule", type=str, default='examples.ScanNetLightningPrecomputed.ScanNetPrecomputed')
        parser.add_argument("--color_backbone", type=str, default=None)
        parser.add_argument("--structure_backbone", type=str, default=None)
        parser.add_argument("--use_wandb", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--log_conf_matrix", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--log_ious", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--save_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--weights", type=str, default=None)
        # parser.add_argument("--use_tb", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser

def get_obj_from_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ME.MinkowskiReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def visualize(pl_module, pl_datamodule):
    from examples.utils import save_pc
    import torch
    # pl_datamodule.__init__()
    pl_datamodule.prepare_data()
    pl_datamodule.setup(stage="validate")
    dataloader = pl_datamodule.val_dataloader()
    data = iter(dataloader).next()
    pl_module = pl_module.cuda()
    data['coords'] = data['coords'].cuda()
    data['feats'] = data['feats'].cuda()
    # data['feats'][:,:3] = torch.rand_like(data['feats'][:,:3]) - 0.5
    # data['feats'][:,3:6] = (torch.rand_like(data['feats'][:,3:6]) * 2) - 1
    # print(data['feats'].min(axis=0), data['feats'].max(axis=0))
    # data['feats'] = torch.rand_like(data['feats'])

    data['feats'].requires_grad_()
    out = pl_module(data)
    if len(out.shape) == 3:
        out = out[0]
    indices = torch.arange(out.shape[0])

    valid_targets = data['labels'] != -1
    
    preds = out.argmax(axis=-1)
    valid_preds = preds[valid_targets]
    preds[~valid_targets] = -1

    # print(preds, preds.shape)
    # import pdb; pdb.set_trace()
    colors = [pl_datamodule.color_map[p] for p in preds.cpu().numpy()]
    save_pc(data['pts'], colors, 'vis.ply')
    colors = [pl_datamodule.color_map[p] for p in data['labels'].cpu().numpy()]
    save_pc(data['pts'], colors, 'vis_gt.ply')

    
    targets = data['labels'].cuda().long()[valid_targets]
    pl_module.val_class_iou(valid_preds, targets)
    iou = pl_module.val_class_iou.compute().cpu().numpy()
    iou = np.array([io for i, io in enumerate(iou) if np.sum(data['labels'].cpu().numpy() == i) > 0])
    miou = iou.mean()

    valid_indices = valid_targets[indices]
    # rand_pt_idx = torch.randint(valid_indices.shape[0], (1,)).item()
    rand_pt_idx = 198832
    rand_pt_class = pl_datamodule.class_labels[data['labels'][rand_pt_idx]]
    print(rand_pt_class, rand_pt_idx, iou, miou)

    # Compute grad with respect to correct class
    correct_label = data['labels'][rand_pt_idx] 
    # pt_out = out[rand_pt_idx, correct_label] - (torch.sum(out[rand_pt_idx, :correct_label]) + torch.sum(out[rand_pt_idx, correct_label+1:]))
    # pt_out.backward(retain_graph=True)
    one_hot_output = torch.FloatTensor(out.size()[-1]).zero_().cuda()
    one_hot_output[correct_label] = 1
    out[rand_pt_idx].backward(retain_graph=True, gradient=one_hot_output)
    
    grad = torch.abs(data['feats'].grad)
    grad = grad / grad.max() * 255.

    # import pdb; pdb.set_trace()
    alpha = 10
    save_pc(data['pts'][rand_pt_idx].reshape(-1,3), np.zeros((1,3)), 'vis_grad_pt_' + rand_pt_class + '_' + str(rand_pt_idx) + '.ply')
    pts_idx = grad[:,:3].norm(dim=-1) > alpha
    save_pc(data['pts'][pts_idx], grad[pts_idx,:3].cpu().numpy(), 'vis_grad_colors_' + rand_pt_class + '_' + str(rand_pt_idx) + '.ply')
    pts_idx = grad[:,3:6].norm(dim=-1) > alpha
    save_pc(data['pts'][pts_idx], grad[pts_idx,3:6].cpu().numpy(), 'vis_grad_normals_' + rand_pt_class + '_' + str(rand_pt_idx) + '.ply')

    grad = data['feats'].grad
    color_grad_norm = grad[:,:3].norm(dim=-1).mean()
    normals_grad_norm = grad[:,3:6].norm(dim=-1).mean()
    print(color_grad_norm, normals_grad_norm)

    # GBP = GuidedBackprop(pretrained_model)
    # guided_grads = GBP.generate_gradients(prep_img, target_class)

    #Grad with respect to incorrect classes
    # pt_out = torch.sum(out[rand_pt_idx, :correct_label]) + torch.sum(out[rand_pt_idx, correct_label+1:])
    # data['feats'].grad = torch.zeros_like(data['feats'].grad)
    # pt_out.backward(retain_graph=True)
    # grad = torch.abs(data['feats'].grad)
    # # grad = ((grad + grad.min()) / (grad.max() + grad.min())) * 255.
    # grad = grad / grad.max() * 255.
    # # save_pc(data['pts'][rand_pt_idx].reshape(-1,3), np.zeros((1,3)), 'vis_neggrad_pt_' + rand_pt_class + '_' + str(rand_pt_idx) + '.ply')
    # pts_idx = grad[:,:3].norm(dim=-1) > alpha
    # save_pc(data['pts'][pts_idx], grad[pts_idx,:3].cpu().numpy(), 'vis_neg_grad_colors_' + rand_pt_class + '_' + str(rand_pt_idx) + '.ply')
    # pts_idx = grad[:,3:6].norm(dim=-1) > alpha
    # save_pc(data['pts'][pts_idx], grad[pts_idx,3:6].cpu().numpy(), 'vis_neg_grad_normals_' + rand_pt_class + '_' + str(rand_pt_idx) + '.ply')
    # import pdb; pdb.set_trace()
    pl_datamodule.teardown(stage="validate")

if __name__ == "__main__":
    
    main_args, args, _ = init_module_from_args(MainArgs)
    seed_everything(main_args.seed, workers=True)

    pl_datamodule = get_obj_from_str(main_args.pl_datamodule)
    pl_datamodule, args, pl_datamodule_args = init_module_from_args(pl_datamodule, args)

    if main_args.color_backbone is not None:
        color_backbone = get_obj_from_str(main_args.color_backbone)
        _, args, color_backbone_args = init_module_from_args(color_backbone, args)
    else:
        color_backbone_args = None

    if main_args.structure_backbone is not None:
        structure_backbone = get_obj_from_str(main_args.structure_backbone)
        _, args, structure_backbone_args = init_module_from_args(structure_backbone, args)
    else:
        structure_backbone_args = None

    pl_module = get_obj_from_str(main_args.pl_module)
    pl_module, args, pl_module_args = init_module_from_args(pl_module, args,
                                                            color_backbone_class=main_args.color_backbone,
                                                            color_backbone_args=color_backbone_args,
                                                            structure_backbone_class=main_args.structure_backbone,
                                                            structure_backbone_args=structure_backbone_args,
                                                            feat_channels=pl_datamodule.feat_channels,
                                                            num_classes=pl_datamodule.NUM_LABELS,
                                                            label_weights=pl_datamodule.labelweights)

    # callbacks = []
    lightning_root_dir = os.path.join('logs', main_args.exp_name, 'train')
    loggers = [TensorBoardLogger(save_dir=lightning_root_dir, name='lightning_logs')]
    os.makedirs(lightning_root_dir, exist_ok=True)
    if main_args.use_wandb:
        tags = ()
        tags += (main_args.pl_module.split('.')[-1],)
        tags += (main_args.pl_datamodule.split('.')[-1],)
        tags += (main_args.run_mode,)
        tags += ("lr_"+str(pl_module.lr),)
        tags += (pl_module.optimizer,)
        tags += (pl_module.scheduler,)
        tags += ("wd_"+str(pl_module.weight_decay),)
        tags += ("seed_"+str(main_args.seed),)
        loggers.append(WandbLogger(save_dir=lightning_root_dir, name=main_args.exp_name, tags=tags))
        


    callbacks = pl_datamodule.callbacks()
    if main_args.log_conf_matrix:
        callbacks.append(ConfusionMatrixPlotCallback())
    if main_args.log_ious:
        callbacks.append(IoUPlotCallback())
    callbacks.append(ModelCheckpoint(monitor='val_miou', mode = 'max', save_top_k=1, save_last=True,
                                    dirpath=os.path.join(lightning_root_dir, loggers[0].name, "version_"+str(loggers[0].version), 'checkpoints')))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    # print(callbacks)

    
    train_dir = os.path.join(lightning_root_dir, 'lightning_logs')
    train_versions = glob.glob(os.path.join(train_dir, '*'))
    resume_from_checkpoint = None
    if main_args.weights is not None:
        pl_module = pl_module.load_from_checkpoint(
                        checkpoint_path=main_args.weights,
                        strict=False,
                        **pl_module_args)
        print(f'Restored {main_args.weights}')
        resume_from_checkpoint = None
    elif len(train_versions) > 0:
        most_recent_train_version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in train_versions])
        most_recent_train_logdir = os.path.join(train_dir, f'version_{most_recent_train_version}')
        print(f'Loading saved model in {most_recent_train_logdir}...')
        ckptdirs = glob.glob(f'{most_recent_train_logdir}/checkpoints/last*')
        if len(ckptdirs) > 0:
            ckpt = ckptdirs[0]
            pl_module = pl_module.load_from_checkpoint(
                        checkpoint_path=ckpt,
                        hparams_file=f'{most_recent_train_logdir}/hparams.yaml',
                        **pl_module_args)
            print(f'Restored {ckpt}')
            resume_from_checkpoint = ckpt

    pl_trainer, args, _ = init_module_from_args(Trainer, args, callbacks=callbacks,
                                             default_root_dir=lightning_root_dir,
                                             plugins=DDPPlugin(find_unused_parameters=False),
                                             resume_from_checkpoint=resume_from_checkpoint,
                                             logger=loggers,
                                             )

    # print("done trainer")
    if pl_trainer.gpus > 1:
        pl_module.convert_sync_batchnorm()

    if main_args.run_mode == 'train':
        pl_trainer.fit(pl_module, pl_datamodule) 
    elif main_args.run_mode == 'validate':
        pl_trainer.validate(pl_module, pl_datamodule)
    elif main_args.run_mode == 'test':
        pl_trainer.test(pl_module, pl_datamodule)
    elif main_args.run_mode == 'visualize':
        visualize(pl_module, pl_datamodule)


