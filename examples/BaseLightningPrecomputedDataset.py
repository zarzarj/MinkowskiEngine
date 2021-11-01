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


class BasePrecomputed(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if name != "self":
                # print(name, value)
                setattr(self, name, value)

        self.feat_channels = 3 * int(self.use_colors) + 3 * int(self.use_normals)
        self.labelweights = None
        self.cache = {}
        if self.use_augmentation:
            if self.load_graph:
                transformations = []
            else:
                transformations = [t.RandomDropout(0.2, 0.2)]
            transformations.extend([
                                      t.ElasticDistortion(0.2, 0.4),
                                      t.ElasticDistortion(0.8, 1.6),
                                      t.RandomScaling(0.9, 1.1),
                                      t.RandomRotation(([-np.pi/64, np.pi/64], [-np.pi/64, np.pi/64], [-np.pi, np.pi])),
                                      t.RandomHorizontalFlip('z'),
                                      t.ChromaticAutoContrast(),
                                      t.ChromaticTranslation(0.1),
                                      t.ChromaticJitter(0.05),
                                    ])
            self.augment = t.Compose(transformations)


    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_files, collate_fn=self.convert_batch,
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_files, collate_fn=self.convert_batch,
                          batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
        return val_dataloader

    def test_dataloader(self):  # Test best validation model once again.
        return DataLoader(self.test_files, collate_fn=self.convert_batch,
                          batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def convert_batch(self, idxs):
        in_dict = self.load_scan_files(idxs)
        for k, v in in_dict.items():
            # print(v)
            if np.all([isinstance(it, torch.Tensor) for it in v]):
                in_dict[k] = torch.cat(v, axis=0)
        return in_dict

    def load_scan_files(self, idxs):
        out_dict = {}
        for batch_idx, scene in enumerate(idxs):
            if self.in_memory and scene in self.cache:
                in_dict = copy.deepcopy(self.cache[scene])
            else:
                in_dict = self.load_sample(scene)
                if self.in_memory:
                    self.cache[scene] = copy.deepcopy(in_dict)
            in_dict['batch_idx'] = batch_idx

            in_dict = self.process_input(in_dict)
            # print(in_dict['labels'].max(), in_dict['labels'].min(), in_dict['labels'].shape)
            # labels = [self.valid_class_ids[label] if label != -1 else -1 for label in in_dict['labels'] ]
            # colors = [self.scannet_color_map[label] for label in labels]
            # save_pc(in_dict['pts'], colors, 'test_pc.ply')
            # assert(True==False)
            for k, v in in_dict.items():
                if scene == idxs[0]:
                    out_dict[k] = [v]
                else:
                    out_dict[k].append(v)
        return out_dict

    def process_input(self, in_dict):
        if self.use_augmentation and self.trainer.training:
            # print(in_dict['colors'][:10])
            # print("augmenting")
            in_dict = self.augment(in_dict)
            # print(in_dict['colors'][:10])
        in_dict['coords'] = in_dict['pts'] / self.voxel_size
        # print(in_dict['pts'])
        in_dict['coords'] = torch.floor(in_dict['coords']).long()
        if self.shift_coords and self.trainer.training:
            in_dict['coords'] += (torch.rand(3) * 100).type_as(in_dict['coords'])
            # print(in_dict['coords'])
        in_dict['coords'] = torch.cat([torch.ones(in_dict['pts'].shape[0], 1).long()*in_dict['batch_idx'], in_dict['coords']], axis=-1)
        if 'colors' in in_dict:
            # print(in_dict['colors'].max())
            in_dict['colors'] = (in_dict['colors'] / 255.) - 0.5
        in_dict['feats'] = self.get_features(in_dict)
        # if self.quantize_input:
            # print(in_dict['coords'], in_dict['feats'], in_dict['labels'])
        # in_dict['coords'], in_dict['feats'], in_dict['labels'] = ME.utils.sparse_quantize(
        #     in_dict['coords'].numpy(), in_dict['feats'].numpy(), labels=in_dict['labels'].long().numpy(), ignore_label=-1)
        # in_dict['coords'], in_dict['feats'], in_dict['labels'] = torch.from_numpy(in_dict['coords']), torch.from_numpy(in_dict['feats']), torch.from_numpy(in_dict['labels'])
        return in_dict

    def get_features(self, in_dict):
        feats = []
        if self.use_colors:
            feats.append(in_dict['colors'])
        if self.use_normals:
            feats.append(in_dict['normals'])
        if len(feats) == 0:
            feats.append(torch.ones((in_dict['pts'].shape[0], 1)))
        out_feats = torch.cat(feats, dim=-1).float()
        return out_feats

    def callbacks(self):
        return []

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ScanNet")
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=6)
        parser.add_argument("--test_batch_size", type=int, default=6)

        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--in_memory", type=str2bool, nargs='?', const=True, default=True)

        parser.add_argument("--structure_feats", type=str, default=None) #"feats_mink"
        parser.add_argument("--color_feats", type=str, default=None) #"feats_pointnet"
        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_normals", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--load_graph", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--use_orig_pcs", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--use_augmentation", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--shift_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--quantize_input", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("--elastic_distortion", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--voxel_size", type=float, default=0.02)
        return parent_parser
