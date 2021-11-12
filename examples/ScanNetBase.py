import os
import glob
from typing import Any, Optional, List, NamedTuple

import torch
import numpy as np
from tqdm import tqdm
from examples.BaseDataset import BaseDataset
from examples.str2bool import str2bool

class ScanNetBase(BaseDataset):
    def __init__(self, **kwargs):
        # print("ScanNetBase init")
        # print(kwargs)
        super().__init__(**kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        # print(kwargs, self.use_orig_pcs)
        self.feat_channels = 3 * int(self.use_colors)  + 3 * int(self.use_normals)
        if self.feat_channels == 0:
            self.feat_channels = 1
        self.scans_dir = os.path.join(self.data_dir, 'scans')
        self.scans_test_dir = os.path.join(self.data_dir, 'scans_test')

        self.class_labels = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')
        self.valid_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
        ]

        self.NUM_LABELS = 150  # Will be converted to 20 as defined in IGNORE_LABELS.
        self.IGNORE_LABELS = tuple(set(range(self.NUM_LABELS)) - set(self.valid_class_ids))
        # map labels not evaluated to ignore_label
        self.label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                self.label_map[l] = -1
            else:
                self.label_map[l] = n_used
                n_used += 1
        self.label_map[-1] = -1
        self.NUM_LABELS -= len(self.IGNORE_LABELS)
        # print("ScanNetBase init done")



    def setup(self, stage: Optional[str] = None):
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
            self.train_files = f.readlines()
            self.train_files = [file[:-5] for file in self.train_files]
        # print(self.train_files)
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
            self.val_files = f.readlines()
            self.val_files = [file[:-5] for file in self.val_files]
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_test.txt'), 'r') as f:
            self.test_files = f.readlines()
            self.test_files = [file[:-5] for file in self.test_files]

    def load_sample(self, idx):
        idx = idx.split('/')[-1].split('.')[0]
        if self.use_orig_pcs:
            scene_data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'scans_processed_full_normals', idx + '.npy')))
            labels = scene_data[:, 9].long()
            labels[labels == 156] = -1
        else:
            scene_data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', idx + '.npy')))
            labels = scene_data[:, 10].long()

        out_dict = {'pts': scene_data[:, :3],  # include xyz by default
                    'labels': labels,
                    'scene_name': idx,
                    } 

        if self.use_colors:
            out_dict['colors'] = scene_data[:, 3:6]

        if self.use_normals:
            out_dict['normals'] = scene_data[:, 6:9]
        return out_dict

    def get_features(self, in_dict):
        feats = []
        if self.use_colors:
            feats.append(in_dict['colors'])
        if self.use_normals:
            feats.append(in_dict['normals'])
        if len(feats) == 0:
            feats.append(torch.ones((in_dict['pts'].shape[0], 1)))
        out_feats = torch.cat(feats, dim=-1).float()
        if self.random_feats:
            out_feats = torch.rand_like(out_feats) - 0.5

        return out_feats

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseDataset.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetBase")
        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_normals", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_orig_pcs", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--random_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--num_classes", type=int, default=20)
        return parent_parser