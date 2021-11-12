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
from examples.utils import interpolate_grid_feats, get_embedder, gather_nd, sparse_collate, save_pc
import MinkowskiEngine as ME

# class SimpleDataset(torch.utils.data.Dataset):


class BasePrecomputed(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if name != "self":
                # print(name, value)
                setattr(self, name, value)
        self.seg_feat_channels = None
        self.kwargs = kwargs
        self.labelweights=None

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_files, collate_fn=self.collate_fn,
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_files, collate_fn=self.collate_fn,
                          batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
        return val_dataloader

    def test_dataloader(self):  # Test best validation model once again.
        return DataLoader(self.test_files, collate_fn=self.collate_fn,
                          batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def collate_fn(self, data):
        data = [self.load_sample(i) for i in data]
        batch_size = len(data)
        out_dict = {}
        for batch_idx, batch in enumerate(data):
            batch['batch_idx'] = batch_idx
            if self.trainer is None:
                batch = self.process_input(batch, training=False)
            else:
                batch = self.process_input(batch, training=self.trainer.training)
            for k, v in batch.items():
                if batch_idx == 0:
                    out_dict[k] = [v]
                else:
                    out_dict[k].append(v)

        for k, v in out_dict.items():
            if np.all([isinstance(it, torch.Tensor) for it in v]):
                if self.dense_input and k != 'labels':
                    # print(v[0].shape)
                    out_dict[k] = torch.stack(v, axis=0)
                    # print(out_dict[k].shape)
                    if k == 'feats':
                        out_dict[k] = out_dict[k].transpose(1,2).contiguous()
                else:
                    out_dict[k] = torch.cat(v, axis=0)
        return out_dict

    def callbacks(self):
        return []

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("BasePrecomputed")
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=6)
        parser.add_argument("--test_batch_size", type=int, default=6)
        parser.add_argument("--num_workers", type=int, default=0)
        return parent_parser
