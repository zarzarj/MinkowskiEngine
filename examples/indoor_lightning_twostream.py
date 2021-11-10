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
        parser.add_argument('--run_mode', type=str, default='train', choices=['train','validate','test'])
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
