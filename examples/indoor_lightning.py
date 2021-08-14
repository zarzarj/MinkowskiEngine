# Copyright (c) NVIDIA Corporation.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
from urllib.request import urlretrieve

from pytorch_lightning import Trainer

from examples.MinkLightning import MinkowskiSegmentationModule
from examples.ScanNetLightning import ScanNet
import MinkowskiEngine as ME

def init_module_from_args(module, args=None):
    parser = argparse.ArgumentParser()
    parser = module.add_argparse_args(parser)
    module_args, args = parser.parse_known_args(args=args)
    return module(**vars(module_args)), args

if __name__ == "__main__":
    pl_module, args = init_module_from_args(MinkowskiSegmentationModule)
    pl_datamodule, args = init_module_from_args(ScanNet, args)
    pl_trainer, args = init_module_from_args(Trainer, args)

    if pl_trainer.gpus > 1:
        pl_module.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(pl_module.model)
    pl_trainer.fit(pl_module, pl_datamodule)
