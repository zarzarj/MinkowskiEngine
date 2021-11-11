import os
import glob
import time
import math
import copy
import inspect
from typing import Any, Optional, List, NamedTuple

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import numpy as np
from tqdm import tqdm
from plyfile import PlyElement, PlyData
import examples.transforms_dict as t
from examples.str2bool import str2bool
from examples.utils import interpolate_grid_feats, get_embedder, gather_nd, sparse_collate, sort_coords
from examples.BaseLightningPrecomputedDataset import BasePrecomputed
from examples.ScanNetBase import ScanNetBase


class ScanNetPrecomputed(ScanNetBase, BasePrecomputed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BasePrecomputed.__init__(self, **kwargs)
        ScanNetBase.__init__(self, **kwargs)
        for name, value in kwargs.items():
            if name != "self":
                # print(name, value)
                setattr(self, name, value)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePrecomputed.add_argparse_args(parent_parser)
        parent_parser = ScanNetBase.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetPrecomputed")
        return parent_parser
