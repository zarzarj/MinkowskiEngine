import os
import io
import argparse
import glob

import numpy as np
from urllib.request import urlretrieve

from pytorch_lightning import Trainer, seed_everything

import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.plugins import DDPPlugin
from examples.str2bool import str2bool

def plot_confusion_matrix(trainer, pl_module, confusion_metric, plot_title):
    tb = pl_module.logger.experiment
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
    tb.add_image(plot_title, im, global_step=pl_module.current_epoch)
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
        parser.add_argument("--pipeline", type=str, default='default', choices=['default', 'implicit', 'revgnn'])
        return parent_parser

if __name__ == "__main__":
    main_args, args, _ = init_module_from_args(MainArgs)
    seed_everything(main_args.seed)
    if main_args.pipeline == 'default':  
        from examples.MinkLightning import MinkowskiSegmentationModule
        from examples.ScanNetLightning import ScanNet
        datamodule, module = ScanNet, MinkowskiSegmentationModule     
    elif main_args.pipeline == 'implicit':
        from examples.MinkLightningLIG import MinkowskiSegmentationModuleLIG
        from examples.ScanNetLightningLIG import ScanNetLIG
        datamodule, module = ScanNetLIG, MinkowskiSegmentationModuleLIG
    elif main_args.pipeline == 'revgnn':
        from examples.rgnn_rooms import RevGNN_Rooms
        from examples.ScanNetLightning import ScanNet
        datamodule, module = ScanNet, RevGNN_Rooms

    pl_module, args, pl_module_args = init_module_from_args(module, args)
    pl_datamodule, args, _ = init_module_from_args(datamodule, args)

    callbacks = []
    callbacks.append(ConfusionMatrixPlotCallback())
    callbacks.append(ModelCheckpoint(monitor='val_miou', mode = 'max', save_top_k=1))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    lightning_root_dir = os.path.join('logs', main_args.exp_name, main_args.run_mode)
    train_dir = os.path.join(lightning_root_dir, '..', 'train', 'lightning_logs')
    train_versions = glob.glob(os.path.join(train_dir, '*'))
    resume_from_checkpoint = None
    if len(train_versions) > 0:
        most_recent_train_version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in train_versions])
        most_recent_train_logdir = os.path.join(train_dir, f'version_{most_recent_train_version}')
        print(f'Loading saved model in {most_recent_train_logdir}...')
        ckptdirs = glob.glob(f'{most_recent_train_logdir}/checkpoints/*')
        if len(ckptdirs) > 0:
            ckpt = ckptdirs[0]
            pl_module = pl_module.load_from_checkpoint(
                        checkpoint_path=ckpt,
                        hparams_file=f'{most_recent_train_logdir}/hparams.yaml',
                        **pl_module_args)
            print(f'Restored {ckpt}')
            resume_from_checkpoint = ckpt
    pl_trainer, args, _ = init_module_from_args(Trainer, args, callbacks=callbacks,
                                             default_root_dir=os.path.join(lightning_root_dir),
                                             plugins=DDPPlugin(find_unused_parameters=False),
                                             resume_from_checkpoint=resume_from_checkpoint)

    if pl_trainer.gpus > 1:
        pl_module.convert_sync_batchnorm()

    if main_args.run_mode == 'train':
        pl_trainer.fit(pl_module, pl_datamodule) 
    elif main_args.run_mode == 'validate':
        pl_trainer.validate(pl_module, pl_datamodule)
    elif main_args.run_mode == 'test':
        pl_trainer.test(pl_module, pl_datamodule)
