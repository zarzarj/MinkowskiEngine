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
# import examples.transforms as t
from examples.str2bool import str2bool
from examples.utils import interpolate_grid_feats, get_embedder, gather_nd, sparse_collate
# from examples.DeepGCN_nopyg.gcn_lib.dense.torch_edge import dense_knn_matrix
# import MinkowskiEngine as ME


class ScanNetPrecomputed(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if name != "self":
                # print(name, value)
                setattr(self, name, value)
        self.scans_dir = os.path.join(self.data_dir, 'scans')
        self.scans_test_dir = os.path.join(self.data_dir, 'scans_test')

        self.class_labels = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')
        self.valid_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
        ]
        self.scannet_color_map = {
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
            40: (100., 85., 144.),
        }

        self.NUM_LABELS = 150  # Will be converted to 20 as defined in IGNORE_LABELS.
        self.IGNORE_LABELS = tuple(set(range(self.NUM_LABELS)) - set(self.valid_class_ids))
        # map labels not evaluated to ignore_label
        self.label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                self.label_map[l] = -100
            else:
                self.label_map[l] = n_used
                n_used += 1
        self.label_map[-100] = -100
        self.NUM_LABELS -= len(self.IGNORE_LABELS)

        self.labelweights = None

        self.cache = {}


    # def prepare_data(self):
    #     if self.load_graph:
    #         all_scans = glob.glob(os.path.join(self.scans_dir, '*')) + glob.glob(os.path.join(self.scans_test_dir, '*'))
    #         os.makedirs(os.path.join(self.data_dir, 'adjs_dense'), exist_ok=True)
    #         for scan in tqdm(all_scans):
    #             scene_name = scan.split('/')[-1]
    #             adj_file = os.path.join(self.data_dir, 'adjs_dense', scene_name + '_adj.pt')
    #             if not os.path.exists(adj_file):
    #                 scan_file = os.path.join(scan, scene_name + '_vh_clean_2.ply')
    #                 with open(scan_file, 'rb') as f:
    #                     plydata = PlyData.read(f)
    #                 pts = torch.from_numpy(np.stack((plydata['vertex']['x'],
    #                                        plydata['vertex']['y'],
    #                                        plydata['vertex']['z'])).T)
    #                 adj = dense_knn_matrix(x=pts.transpose(0,1).unsqueeze(0).unsqueeze(-1), k=16)
    #                 torch.save(adj, adj_file)

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
        # print(torch.stack(in_dict['color_feats']).shape)
        for k, v in in_dict.items():
            if np.all([it is not None for it in v]):
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

            in_dict = self.process_input(in_dict)
            in_dict['batch_idx'] = torch.ones(in_dict['pts'].shape[0]) * batch_idx
            for k, v in in_dict.items():
                if scene == idxs[0]:
                    out_dict[k] = [v]
                else:
                    out_dict[k].append(v)
        return out_dict

    def load_sample(self, idx):
        if self.use_orig_pcs:
            scene_data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'scans_processed_full_normals', idx + '.npy')))
            labels = scene_data[:, 9].long()
            labels[labels == 156] = -100
        else:
            scene_data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', idx + '.npy')))
            labels = scene_data[:, 10].long()

        out_dict = {'pts': scene_data[:, :3],  # include xyz by default
                    'labels': labels,
                    } 

        if self.use_colors:
            out_dict['colors'] = scene_data[:, 3:6]

        if self.use_normals:
            out_dict['normals'] = scene_data[:, 6:9]

        if self.structure_feats is not None:
            out_dict['structure_feats'] = torch.load(os.path.join(self.data_dir, self.structure_feats, idx + '_feats.pt'))
        else:
            out_dict['structure_feats'] = None

        if self.color_feats is not None:
            out_dict['color_feats'] = torch.load(os.path.join(self.data_dir, self.color_feats, idx + '_feats.pt'))
        else:
            out_dict['color_feats'] = None

        if self.load_graph:
            out_dict['adj'] = torch.load(os.path.join(self.data_dir, 'adjs_dense', idx + '_adj.pt'))

        return out_dict

    def process_input(self, in_dict):
        if self.use_colors:
            in_dict['colors'] /= 255. # normalize the rgb values to [0, 1]
        return in_dict

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

        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_normals", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--load_graph", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--use_orig_pcs", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser
