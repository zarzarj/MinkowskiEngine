import os
import glob
import time
import math
import copy
import inspect
from typing import Any, Optional, List, NamedTuple

import numpy as np
from tqdm import tqdm
import examples.transforms_dict as t
from examples.str2bool import str2bool
from examples.BaseLightningPointNet import BasePointNetLightning, BaseWholeScene, BaseChunked
from examples.ScanNetBase import ScanNetBase


class ScanNetPointNet(ScanNetBase, BasePointNetLightning):
    def __init__(self, **kwargs):
        # print("ScanNetPointNet init")
        BasePointNetLightning.__init__(self, **kwargs)
        # print(self.kwargs)
        ScanNetBase.__init__(self, **kwargs)
        self.kwargs = copy.deepcopy(kwargs)
        self.whole_scene_dataset = ScanNetWholeScene
        self.chunked_scene_dataset = ScanNetChunked
        # print("ScanNetPointNet init done")

    def setup(self, stage: Optional[str] = None):
        # print(self.kwargs)
        ScanNetBase.setup(self, stage)
        BasePointNetLightning.setup(self, stage)
        
    #     if self.use_whole_scene:
    #         self.train_dataset = ScanNetWholeScene(phase="train", scene_list=self.train_files, **self.kwargs)
    #         self.val_dataset = ScanNetWholeScene(phase="val", scene_list=self.val_files, **self.kwargs)
    #     else:
    #         self.train_dataset = ScanNetChunked(phase="train", scene_list=self.train_files, **self.kwargs)
    #         self.val_dataset = ScanNetChunked(phase="val", scene_list=self.val_files, **self.kwargs)
        # self.val_dataset.generate_chunks()

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePointNetLightning.add_argparse_args(parent_parser)
        parent_parser = ScanNetBase.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetPointNet")
        return parent_parser


class ScanNetWholeScene(ScanNetBase, BaseWholeScene):
    def __init__(self, **kwargs):
        # print("ScanNetWholeScene init")
        BaseWholeScene.__init__(self, **kwargs)
        ScanNetBase.__init__(self, **kwargs)
        # print("ScanNetWholeScene init done")

class ScanNetChunked(ScanNetBase, BaseChunked):
    def __init__(self, **kwargs):
        # print("ScanNetChunked init")
        BaseChunked.__init__(self, **kwargs)
        ScanNetBase.__init__(self, **kwargs)
        # print("ScanNetChunked init done")