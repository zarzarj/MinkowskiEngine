import os
import sys
import glob
import time
import math
import copy
import inspect
import torch
import numpy as np

from typing import Any, Optional, List, NamedTuple
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from examples.str2bool import str2bool

from tqdm import tqdm
from pytorch_lightning.callbacks import Callback

class BaseDataset():
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)  
        if self.use_augmentation:
            self.augment = self.create_augs()
        if self.to_hsv:
            self.color_transform = t.RGBtoHSV()

    def process_input(self, in_dict, training=False):
        if self.to_hsv:
            in_dict = self.color_transform(in_dict)
        if self.use_augmentation and training:
            in_dict = self.augment(in_dict)
        in_dict['num_pts'] = in_dict['pts'].shape[0]
        in_dict['coords'] = in_dict['pts'] / self.voxel_size
        if self.shift_coords and training:
            in_dict['coords'] += (torch.rand(3) * 100).type_as(in_dict['coords'])
        if self.batch_fusion and self.trainer.training:
            in_dict['batch_idx'] = int(in_dict['batch_idx'] / 2)

        in_dict['coords'] = torch.cat([torch.ones(in_dict['coords'].shape[0], 1).long()*in_dict['batch_idx'], in_dict['coords']], axis=-1)

        if 'colors' in in_dict:
            in_dict['colors'] = (in_dict['colors'] / 255.) - 0.5
            if self.rand_colors:
                in_dict['colors'] = torch.rand_like(in_dict['colors']) - 0.5
        in_dict['feats'] = self.get_features(in_dict)
        return in_dict

    def create_augs(self, m=1.0):
        transformations = []
        if self.point_dropout:
            transformations = [t.RandomDropout(0.2 * m)]
        if self.color_aug:
            transformations.extend([
                                  t.ChromaticAutoContrast(),
                                  t.ChromaticTranslation(0.1 * m),
                                  t.ChromaticJitter(0.05 * m),
                                  ])
        if self.structure_aug:
            transformations.extend([
                                  t.ElasticDistortion(0.2 * m, 0.4 * m),
                                  t.ElasticDistortion(0.8 * m, 1.6 * m),
                                  t.RandomScaling(0.9 / m, 1.1 * m),
                                  t.RandomRotation(([-np.pi/64 * m, np.pi/64 * m], [-np.pi/64 * m, np.pi/64 * m], [-np.pi, np.pi])),
                                  t.RandomHorizontalFlip('z'),
                                ])
            if self.pos_jitter:
                transformations.append(t.PositionJitter(0.005 * m))
        return t.Compose(transformations)

    def update_aug(self, m=1.0):
        print("Setting aug multiplier to: ", m)
        self.augment = self.create_augs(m)

    def collate_fn(self, data):
        batch_size = len(data)
        # batch_idx = torch.cat([torch.ones(data[i]['pts'].shape[0], 1) * i for i in range(batch_size)], axis=0)
        out_dict = {}
        for batch_idx, batch in enumerate(data):
            batch['batch_idx'] = batch_idx
            batch = self.process_input(batch)
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
                # print(k, out_dict[k].shape)
        # print(out_dict)
        # for k, v in 
        # assert(True == False)
        return out_dict

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseDataset")
        parser.add_argument("--use_augmentation", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--to_hsv", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--pos_jitter", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--color_aug", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--structure_aug", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--rand_colors", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--shift_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--point_dropout", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--batch_fusion", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--dense_input", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser