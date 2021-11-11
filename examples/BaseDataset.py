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
import examples.transforms_dict as t

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
        self.color_map = {
            0: (0., 0., 0.),
            1: (174., 199., 232.),
            2: (152., 223., 138.),
            3: (31., 119., 180.),
            4: (255., 187., 120.),
            5: (188., 189., 34.),
            6: (140., 86., 75.),
            7: (255., 152., 150.),
            8: (214., 39., 40.),
            9: (197., 176., 213.),
            10: (148., 103., 189.),
            11: (196., 156., 148.),
            12: (23., 190., 207.),
            13: (100., 85., 144.),
            14: (247., 182., 210.),
            15: (66., 188., 102.),
            16: (219., 219., 141.),
            17: (140., 57., 197.),
            18: (202., 185., 52.),
            19: (51., 176., 203.),
            20: (200., 54., 131.),
            21: (92., 193., 61.),
            22: (78., 71., 183.),
            23: (172., 114., 82.),
            24: (255., 127., 14.),
            25: (91., 163., 138.),
            26: (153., 98., 156.),
            27: (140., 153., 101.),
            28: (158., 218., 229.),
            29: (100., 125., 154.),
            30: (178., 127., 135.),
            32: (146., 111., 194.),
            33: (44., 160., 44.),
            34: (112., 128., 144.),
            35: (96., 207., 209.),
            36: (227., 119., 194.),
            37: (213., 92., 176.),
            38: (94., 106., 211.),
            39: (82., 84., 163.),
            # 40: (100., 85., 144.),
            -1: (255., 0., 0.),
        }

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

    # def __getitem__(self, index):
    #     start = time.time()
    #     print(index)
    #     in_dict = self.load_sample(index)
    #     fetch_time = time.time() - start
    #     return in_dict

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseDataset")
        parser.add_argument("--data_dir", type=str, default=None)
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
        parser.add_argument("--voxel_size", type=float, default=0.02)
        return parent_parser