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

    # def prepare_data(self):
    #     super().prepare_data()
    #     if self.load_graph:
    #         import torch_geometric
    #         adjs_path = os.path.join(self.preprocess_path, 'adjs')
    #         os.makedirs(adjs_path, exist_ok=True)
    #         for room_folder in tqdm(all_rooms):
    #             room_name = room_folder.split('/')[-1]
    #             area_name = room_folder.split('/')[-2]
    #             room_files = glob.glob(os.path.join(self.preprocess_path, area_name + '_' + room_name + '*.pt'))
    #             for room_file in room_files:
    #                 room_adj_file =  os.path.join('/', *room_file.split('/')[:-1], 'adjs', room_file.split('/')[-1][:-3] + '_adj.pt')
    #                 if not os.path.exists(room_adj_file):
    #                     room = torch.load(room_file)
    #                     pts = room[:,:3]
    #                     adj = torch_geometric.nn.pool.knn_graph(x=pts, k=16,
    #                                                     loop=False, flow='source_to_target',
    #                                                     cosine=False)
    #                     torch.save(adj, room_adj_file)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePrecomputed.add_argparse_args(parent_parser)
        parent_parser = S3DISBase.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("S3DISPrecomputed")
        return parent_parser
