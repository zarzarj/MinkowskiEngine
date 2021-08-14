import os
import io
import argparse
import glob

import numpy as np
from urllib.request import urlretrieve

from pytorch_lightning import Trainer

from examples.MinkLightning import MinkowskiSegmentationModule
from examples.ScanNetLightning import ScanNet
import MinkowskiEngine as ME

import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_lightning.callbacks import ModelCheckpoint, Callback



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
        plot_confusion_matrix(trainer, pl_module, pl_module.train_metrics["ConfusionMatrix"], "train_confusion_matrix")
        plot_confusion_matrix(trainer, pl_module, pl_module.train_metrics["NormalizedConfusionMatrix"], "train_normalized_confusion_matrix")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        plot_confusion_matrix(trainer, pl_module, pl_module.val_metrics["ConfusionMatrix"], "val_confusion_matrix")
        plot_confusion_matrix(trainer, pl_module, pl_module.val_metrics["NormalizedConfusionMatrix"], "val_normalized_confusion_matrix")

def init_module_from_args(module, args=None, **kwargs):
    parser = argparse.ArgumentParser()
    parser = module.add_argparse_args(parser)
    module_args, args = parser.parse_known_args(args=args)
    args_dict = vars(module_args)
    args_dict.update(kwargs)
    return module(**args_dict), args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='default')
    main_args, args = parser.parse_known_args()

    pl_module, args = init_module_from_args(MinkowskiSegmentationModule)
    pl_datamodule, args = init_module_from_args(ScanNet, args)

    callbacks = []
    callbacks.append(ConfusionMatrixPlotCallback())
    callbacks.append(ModelCheckpoint(monitor='val_miou', mode = 'max', save_top_k=1))
    lightning_root_dir = f'logs/{main_args.exp_name}/'
    pl_trainer, args = init_module_from_args(Trainer, args, callbacks=callbacks,
                                             default_root_dir=lightning_root_dir)

    dirs = glob.glob(os.path.join(lightning_root_dir, 'lightning_logs', '*'))
    if len(dirs) > 0:
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = os.path.join(lightning_root_dir, 'lightning_logs', f'version_{version}')
        print(f'Loading saved model in {logdir}...')
        ckptdirs = glob.glob(f'{logdir}/checkpoints/*')
        if len(ckptdirs) > 0:
            ckpt = ckptdirs[0]
            pl_module = pl_module.load_from_checkpoint(
                        checkpoint_path=ckpt,
                        hparams_file=f'{logdir}/hparams.yaml')
            print(f'Restored {ckpt}')
        else:
            print('No model found!')
            print('Training from scratch...')

    if pl_trainer.gpus > 1:
        pl_module.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(pl_module.model)
    pl_trainer.fit(pl_module, pl_datamodule)
