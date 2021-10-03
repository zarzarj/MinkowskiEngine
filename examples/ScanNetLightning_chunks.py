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

# import MinkowskiEngine as ME

# from examples.voxelizer import SparseVoxelizer
# import examples.transforms as t
from examples.str2bool import str2bool
from examples.ScanNetLightning import ScanNet
from examples.utils import gather_nd, sparse_collate
# from examples.utils import get_embedder

class ScanNet_chunks(ScanNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_ply(self, idx):
        out_dict = super().load_ply(idx)
        # print("training: ", self.trainer.training)
        # print("validating: ", self.trainer.validating)
        if self.trainer.training:
            pts = out_dict['pts']
            coordmax = torch.max(pts, axis=0)[0]
            coordmin = torch.min(pts, axis=0)[0]
            curcenter = pts[torch.randint(high=pts.shape[0], size=(1,))[0]]
            curmin = curcenter-self.chunk_size
            curmax = curcenter+self.chunk_size
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            chunk_idx = torch.sum((pts>=(curmin-0.2))*(pts<=(curmax+0.2)),axis=1)==3
            for k, v in out_dict.items():
                if k != 'scene_name':
                    out_dict[k] = out_dict[k][chunk_idx]
        else:
            print("not training")
        return out_dict

    def process_input(self, input_dict):
        if self.trainer.training and self.max_num_pts > 0 and self.max_num_pts < input_dict['pts'].shape[0]:
            perm = torch.randperm(input_dict['pts'].shape[0])[:self.max_num_pts]
        else:
            perm = torch.arange(input_dict['pts'].shape[0])
            
        if self.permute_points:
            perm = perm[torch.randperm(perm.shape[0])]

        input_dict['pts'] = input_dict['pts'][perm]
        input_dict['colors'] = input_dict['colors'][perm]
        input_dict['labels'] = input_dict['labels'][perm]
        if self.use_implicit_feats:
            input_dict['implicit_feats'] = input_dict['implicit_feats'][perm]
        input_dict['colors'] = (input_dict['colors'] / 255.) - 0.5
        input_dict['feats'] = self.get_features(input_dict)
        input_dict['coords'] = input_dict['pts'] / self.voxel_size
        input_dict['seg_feats'] = None
        input_dict['rand_shift'] = None
        return input_dict

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        coords_batch, feats_batch = sparse_collate(input_dict['coords'], input_dict['feats'],
                                                                          dtype=torch.float32)
        update_dict = {"coords": coords_batch,
                    "feats": feats_batch,
                    "idxs": idxs}
        input_dict.update(update_dict)
        return input_dict

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            coords = self._translate(coords)
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            coords = self._translate(coords)
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            coords = self._rotate(coords)
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            coords = self._translate(coords)
            coords = self._rotate(coords)
            coords = self._scale(coords)

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords
        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]
        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T
        point_set[:, :3] = coords
        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]
        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords
        return point_set

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = ScanNet.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNet_chunks")
        parser.add_argument("--chunk_size", type=float, default=0.75)
        return parent_parser