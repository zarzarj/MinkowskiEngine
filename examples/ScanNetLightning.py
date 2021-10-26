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

# import MinkowskiEngine as ME


class ScanNet(LightningDataModule):
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

        self.feat_channels = self.implicit_feat_channels * int(self.use_implicit_feats) \
                           + 3 * int(self.use_colors) + 3 * int(self.use_coords) \
                           + 30 * int(self.use_coord_pos_encoding)
        if self.feat_channels == 0:
            self.feat_channels = 1
        self.seg_feat_channels = self.feat_channels + 3
        self.labelweights=None

    def prepare_data(self):
        if self.save_preds:
            os.makedirs(os.path.join(self.data_dir, 'output'), exist_ok=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage =='validate':
            with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
                self.train_files = f.readlines()
                self.train_files = [file[:-5] for file in self.train_files]
            # print(self.train_files)
            with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
                self.val_files = f.readlines()
                self.val_files = [file[:-5] for file in self.val_files]
        else:
            with open(os.path.join(self.data_dir, 'splits', 'scannetv2_test.txt'), 'r') as f:
                self.test_files = f.readlines()
                self.test_files = [file[:-5] for file in self.test_files]
        if self.use_coord_pos_encoding:
            self.embedder, _ = get_embedder(self.coord_pos_encoding_multires)
        if self.in_memory:
            self.cache = {}

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
        input_dict = self.load_scan_files(idxs)
        coords_batch, feats_batch, labels_batch = sparse_collate(input_dict['coords'], input_dict['feats'], input_dict['labels'],
                                                                          dtype=torch.float32)
        return {"coords": coords_batch,
                "feats": feats_batch,
                "seg_feats": input_dict['seg_feats'],
                "labels": labels_batch.long(),
                "idxs": idxs,
                "pts": input_dict['pts'],
                "rand_shift": input_dict['rand_shift'],
                }

    def load_scan_files(self, idxs):
        out_dict = {}
        
        for i in idxs:
            if self.in_memory and i in self.cache:
                input_dict = copy.deepcopy(self.cache[i])
            else:
                input_dict = self.load_ply(i)
                if self.in_memory:
                    # print(i, i in self.cache)
                    self.cache[i] = copy.deepcopy(input_dict)
                    # print(self.cache.keys())
            input_dict = self.process_input(input_dict)
            for k, v in input_dict.items():
                if i == idxs[0]:
                    out_dict[k] = [v]
                else:
                    out_dict[k].append(v)
        return out_dict

    def load_ply_file(self, file_name, load_labels=False):
        with open(file_name, 'rb') as f:
            plydata = PlyData.read(f)
        pts = torch.from_numpy(np.stack((plydata['vertex']['x'],
                               plydata['vertex']['y'],
                               plydata['vertex']['z'])).T)

        pts_min = pts.min(axis=0)[0]
        pts -= pts_min
        colors = torch.from_numpy(np.stack((plydata['vertex']['red'],
                           plydata['vertex']['green'],
                           plydata['vertex']['blue'])).T)
        if load_labels:
            label_file = file_name[:-4] + '.labels.ply'
            with open(label_file, 'rb') as f:
                plydata = PlyData.read(f)
            labels = np.array(plydata['vertex']['label'], dtype=np.uint8)
            labels = torch.tensor([self.label_map[x] for x in labels], dtype=torch.long)
        else:
            labels = torch.zeros(pts.shape[0])

        if self.trainer.training and self.resample_mesh:
            faces = torch.from_numpy(np.stack(plydata['face']['vertex_indices'])).long()
            face_pts = pts[faces]
            bary_pts = torch.rand((faces.shape[0], 2))
            bary_pts[bary_pts.sum(axis=1) > 1] /= 2
            pts = face_pts[:, 0, :] +                                        \
                            (face_pts[:, 1, :] - face_pts[:, 0, :]) * bary_pts[:, 0].unsqueeze(1) + \
                            (face_pts[:, 2, :] - face_pts[:, 0, :]) * bary_pts[:, 1].unsqueeze(1)
            face_colors = colors[faces].float()
            colors = face_colors[:, 0, :] +                                        \
                              (face_colors[:, 1, :] - face_colors[:, 0, :]) * bary_pts[:,0].unsqueeze(1) + \
                              (face_colors[:, 2, :] - face_colors[:, 0, :]) * bary_pts[:,1].unsqueeze(1)
            labels = labels[faces, 0]
            if self.keep_same_labels:
                valid_face_labels = torch.all(face_labels == face_labels[:,0].unsqueeze(1), axis=1)
                pts = pts[valid_face_labels]
                colors = colors[valid_face_labels]
                labels = labels[valid_face_labels]
            # print("resampling mesh")
        return pts, colors, labels

    def load_ply(self, idx):
        # scan_file = self.scan_files[idx]
        # print("idx", idx)
        scan_file = os.path.join(self.data_dir, 'scans', idx, idx + '_vh_clean_2.ply')
        pts, colors, labels = self.load_ply_file(scan_file,
                                                    load_labels=(self.trainer.training
                                                                 or self.trainer.validating)
                                                    )
        out_dict = {'pts': pts,
                    'colors': colors,
                    'labels': labels,
                    'scene_name': idx,
                    }
        if self.use_implicit_feats:
            implicit_feats = self.load_implicit_feats(scan_file, pts)
            out_dict['implicit_feats'] = implicit_feats
        # self.samples[idx] = copy.deepcopy(out_dict)
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

        input_dict['colors'] = (input_dict['colors'] / 255.) - 0.5
        input_dict['coords'] = input_dict['pts'] / self.voxel_size
        input_dict['coords'] = torch.floor(input_dict['coords']).long()
        if self.shift_coords and self.trainer.training:
            input_dict['rand_shift'] = (torch.rand(3) * 100).type_as(input_dict['coords'])
            input_dict['coords'] += input_dict['rand_shift']
        else:
            input_dict['rand_shift'] = None
        
        input_dict['feats'] = self.get_features(input_dict)
        # print(input_dict['coords'].shape, input_dict['feats'].shape)
        input_dict['seg_feats'] = None
        # del input_dict['pts']
        return input_dict

    def get_features(self, input_dict):
        feats = []
        if self.use_colors:
            feats.append(input_dict['colors'])
        if self.use_coords:
            feats.append(input_dict['pts'])
        if self.use_implicit_feats:
            feats.append(input_dict['implicit_feats'])
        if self.use_coord_pos_encoding:
            feats.append(self.embedder(input_dict['pts']))
        if len(feats) == 0:
            feats.append(torch.ones((input_dict['pts'].shape[0], 1)))
        # for feat in feats:
        #     print(feat.shape)
        out_feats = torch.cat(feats, dim=-1)
        # print(out_feats.shape)
        return out_feats


    def load_implicit_feats(self, file_name, pts):
        scene_name = file_name.split('/')[-2]
        lats_file = os.path.join(self.data_dir, 'lats', scene_name + self.lats_file_suffix + '.npy')
        grid = torch.from_numpy(np.load(lats_file))
        lat, xloc, weights = interpolate_grid_feats(pts, grid, overlap_factor=self.overlap_factor)
        if self.interp_grid_feats:
            implicit_feats = torch.bmm(weights.unsqueeze(dim=1), lat).squeeze(1)
        else:
            implicit_feats = torch.cat([lat, xloc], dim=-1).reshape(lat.shape[0], -1)

        return implicit_feats

    def callbacks(self):
        return []

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ScanNet")
        parser.add_argument("--num_classes", type=int, default=20)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=6)
        parser.add_argument("--test_batch_size", type=int, default=6)
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--save_preds", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--train_percent", type=float, default=0.8)
        parser.add_argument("--train_subset", type=float, default=1.0)
        parser.add_argument("--interp_grid_feats", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--overlap_factor", type=int, default=2)
        parser.add_argument("--lats_file_suffix", type=str, default='-d1e-05-vertices-st20000')

        parser.add_argument("--point_subsampling_percent", type=float, default=1.0)
        parser.add_argument("--voxel_size", type=float, default=0.02)
        parser.add_argument("--use_implicit_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--implicit_feat_channels", type=int, default=32)
        parser.add_argument("--use_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_coord_pos_encoding", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--coord_pos_encoding_multires", type=int, default=10)
        parser.add_argument("--shift_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--permute_points", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--resample_mesh", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--max_num_pts", type=int, default=-1)

        parser.add_argument("--in_memory", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--keep_same_labels", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--rand_feats", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser
