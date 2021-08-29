import os
import glob
import time
import math
import inspect
from typing import Any, Optional, List, NamedTuple

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import numpy as np
from tqdm import tqdm
from plyfile import PlyElement, PlyData

import MinkowskiEngine as ME

from examples.voxelizer import SparseVoxelizer
import examples.transforms as t
from examples.str2bool import str2bool
from examples.ScanNetLightning import ScanNet
# from examples.utils import get_embedder

class ScanNetLIG(ScanNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = ScanNet.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetLIG")
        return parent_parser

    def process_input(self, input_dict):
        # print(input_dict)
        input_dict['coords'], input_dict['lats'] = input_dict['implicit_feats']
        if self.shift_coords and self.trainer.training:
            input_dict['rand_shift'] = (torch.rand(3) * 100).type_as(input_dict['coords'])
            input_dict['coords'] += input_dict['rand_shift']
        del input_dict['implicit_feats']
        input_dict['feats'] = self.get_features(input_dict)
        return input_dict

    def load_implicit_feats(self, file_name, pts):
        scene_name = file_name.split('/')[-2]
        mask_file = os.path.join(self.data_dir, 'masks', scene_name+'-d1e-05-ps0.npy')
        lats_file = os.path.join(self.data_dir, 'lats', scene_name+'-d1e-05-ps0.npy')
        mask = torch.from_numpy(np.load(mask_file))
        lats = torch.from_numpy(np.load(lats_file))

        grid_range = [torch.arange(s) for s in mask.shape]
        grid =  torch.stack(torch.meshgrid(grid_range), dim=-1)
        # print(mask_file, grid.shape)
        coords = grid[mask]
        # print(mask_file, grid.shape, coords.max(dim=0)[0], coords.min(dim=0)[0], pts.max(dim=0)[0], pts.min(dim=0)[0])
        # sptensor = ME.SparseTensor(features=lats, coordinates=mask)
        return (coords, lats)

    def get_features(self, input_dict):
        feats = []
        if self.use_colors:
            feats.append(input_dict['colors'])
        if self.use_coords:
            feats.append(input_dict['pts'])
        if self.use_coord_pos_encoding:
            feats.append(self.embedder(input_dict['pts']))
        if len(feats) == 0:
            return None
        out_feats = torch.cat(feats, dim=-1)
        return out_feats

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        coords_batch, lats_batch = ME.utils.sparse_collate(input_dict['coords'],
                                                            input_dict['lats'],
                                                            dtype=torch.float32)
        out_dict = {"coords": coords_batch,
                    "lats": lats_batch,
                    "feats": input_dict['feats'],
                    "pts": input_dict['pts'],
                    "labels": input_dict['labels'],
                    "idxs": idxs,
                    }
        if self.shift_coords and self.trainer.training:
            out_dict["rand_shift"] = input_dict['rand_shift']
        return out_dict