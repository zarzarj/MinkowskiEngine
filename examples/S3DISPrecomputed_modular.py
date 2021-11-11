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
from examples.BaseLightningPrecomputedDataset import BasePrecomputed
from examples.S3DISBase import S3DISBase


class S3DISPrecomputed(S3DISBase, BasePrecomputed):
    def __init__(self, **kwargs):
        BasePrecomputed.__init__(self, **kwargs)
        S3DISBase.__init__(self, **kwargs)
        for name, value in kwargs.items():
            if name != "self":
                # print(name, value)
                setattr(self, name, value)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePrecomputed.add_argparse_args(parent_parser)
        parent_parser = S3DISBase.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("S3DISPrecomputed")
        return parent_parser
