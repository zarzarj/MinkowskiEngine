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
import examples.transforms as t
from examples.str2bool import str2bool
from examples.utils import interpolate_grid_feats, get_embedder

import MinkowskiEngine as ME

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

    def prepare_data(self):
        if self.save_preds:
            os.makedirs(os.path.join(self.data_dir, 'output'), exist_ok=True)
        train_filename = os.path.join(self.data_dir, 'train_idx.npy')
        val_filename = os.path.join(self.data_dir, 'val_idx.npy')
        if not os.path.exists(train_filename) or not os.path.exists(val_filename):
            scan_files = glob.glob(os.path.join(self.scans_dir, '*'))
            idxs = np.random.permutation(len(scan_files))
            num_training = math.ceil(idxs.shape[0] * self.train_percent)
            train_idx, val_idx = idxs[:num_training], idxs[num_training:]
            np.save(train_filename, train_idx)
            np.save(val_filename, val_idx)
        # scan_test_filenames = glob.glob(os.path.join(self.scans_test_dir, '*'))

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage =='validate':
            self.scan_files = glob.glob(os.path.join(self.scans_dir, '*', '*_vh_clean_2.ply'))
            self.train_idx = torch.from_numpy(np.load(os.path.join(self.data_dir, 'train_idx.npy')))
            self.val_idx = torch.from_numpy(np.load(os.path.join(self.data_dir, 'val_idx.npy')))
        else:
            self.scan_files = glob.glob(os.path.join(self.scans_test_dir, '*', '_vh_clean_2.ply'))
            self.test_idx = torch.from_numpy(np.arange(len(self.scan_files)))
        self.scan_files.sort()
        # print(self.scan_files[97])
        if self.use_coord_pos_encoding:
            self.embedder, _ = get_embedder(self.coord_pos_encoding_multires)
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_idx, collate_fn=self.convert_batch,
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_idx, collate_fn=self.convert_batch,
                          batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
        return val_dataloader

    def test_dataloader(self):  # Test best validation model once again.
        return DataLoader(self.test_idx, collate_fn=self.convert_batch,
                          batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        feats = self.get_features(input_dict)
        coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(input_dict['coords'],
                                                                          feats, input_dict['labels'],
                                                                          dtype=torch.float32)
        return {"coords": coords_batch,
                "feats": feats_batch,
                "labels": labels_batch.long(),
                "idxs": idxs,
                }

    def load_scan_files(self, idxs):
        out_dict = {}
        for i in idxs:
            input_dict = self.load_ply(i)
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
            resampled_pts = face_pts[:, 0, :] +                                        \
                            (face_pts[:, 1, :] - face_pts[:, 0, :]) * bary_pts[:, 0].unsqueeze(1) + \
                            (face_pts[:, 2, :] - face_pts[:, 0, :]) * bary_pts[:, 1].unsqueeze(1)
            face_colors = colors[faces].float()
            resampled_colors = face_colors[:, 0, :] +                                        \
                              (face_colors[:, 1, :] - face_colors[:, 0, :]) * bary_pts[:,0].unsqueeze(1) + \
                              (face_colors[:, 2, :] - face_colors[:, 0, :]) * bary_pts[:,1].unsqueeze(1)
            face_labels = labels[faces]
            valid_face_labels = torch.all(face_labels == face_labels[:,0].unsqueeze(1), axis=1)
            resampled_labels = face_labels[:, 0]

            pts = resampled_pts[valid_face_labels]
            colors = resampled_colors[valid_face_labels]
            labels = resampled_labels[valid_face_labels]

        if self.trainer.training and self.max_num_pts > 0 and self.max_num_pts < pts.shape[0]:
            subsample_idx = torch.randperm(pts.shape[0])[:self.max_num_pts]
            pts = pts[subsample_idx]
            colors = colors[subsample_idx]
            labels = labels[subsample_idx]
        return pts, colors, labels

    def load_ply(self, idx):
        scan_file = self.scan_files[idx]
        pts, colors, labels = self.load_ply_file(scan_file,
                                                    load_labels=(self.trainer.training
                                                                 or self.trainer.validating)
                                                    )
        out_dict = {'pts': pts,
                    'colors': colors,
                    'labels': labels,
                    }
        if self.use_implicit_feats:
            implicit_feats = self.load_implicit_feats(scan_file, pts)
            out_dict['implicit_feats'] = implicit_feats
        # self.samples[idx] = copy.deepcopy(out_dict)
        return out_dict

    def process_input(self, input_dict):
        if self.permute_points:
            perm = torch.randperm(input_dict['coords'].shape[0])
            for k, v in input_dict.items():
                input_dict[k] = v[perm]

        input_dict['coords'] = input_dict['pts'] / self.voxel_size
        if self.shift_coords and self.trainer.training:
            input_dict['coords'] += (torch.rand(3) * 100).type_as(input_dict['coords'])
        input_dict['colors'] = (input_dict['colors'] / 255.) - 0.5
        del input_dict['pts']
        return input_dict

    def get_features(self, input_dict):
        feats = []
        if self.use_colors:
            feats.append(input_dict['colors'])
        if self.use_coords:
            feats.append(input_dict['coords'])
        if self.use_implicit_feats:
            feats.append(input_dict['implicit_feats'])
        if self.use_coord_pos_encoding:
            feats.append([self.embedder(coord) for coord in input_dict['coords']])
        if len(feats) == 0:
            feats.append([torch.ones((coords.shape[0], 1)) for coords in input_dict['coords']])
        out_feats = []
        for i in range(len(feats[0])):
            cur_all_feats = [feat[i] for feat in feats]
            # print(cur_all_feats, cur_a)
            out_feats.append(torch.cat(cur_all_feats, dim=-1))
        return out_feats


    def load_implicit_feats(self, file_name, pts):
        scene_name = file_name.split('/')[-2]
        implicit_feat_file = os.path.join(self.data_dir, 'implicit_feats', scene_name+'-d1e-05-ps0.pt')
        if not os.path.exists(implicit_feat_file):
            os.makedirs(os.path.join(self.data_dir, 'implicit_feats'), exist_ok=True)
            mask_file = os.path.join(self.data_dir, 'masks', scene_name+'-d1e-05-ps0.npy')
            lats_file = os.path.join(self.data_dir, 'lats', scene_name+'-d1e-05-ps0.npy')
            mask = torch.from_numpy(np.load(mask_file)).bool()
            lats = torch.from_numpy(np.load(lats_file))
            grid = torch.zeros(mask.shape + (lats.shape[-1],), dtype=torch.float32)
            grid[mask] = lats
            lat, xloc = interpolate_grid_feats(pts, grid)
            implicit_feats = torch.cat([lat, xloc], dim=-1)
            torch.save(implicit_feats, implicit_feat_file)
        else:
            implicit_feats = torch.load(implicit_feat_file)
        return implicit_feats

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ScanNet")
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=6)
        parser.add_argument("--test_batch_size", type=int, default=6)
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--save_preds", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--train_percent", type=float, default=0.8)
        parser.add_argument("--voxel_size", type=float, default=0.02)
        parser.add_argument("--use_implicit_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_coord_pos_encoding", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--coord_pos_encoding_multires", type=int, default=10)
        parser.add_argument("--shift_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--permute_points", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--resample_mesh", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--max_num_pts", type=int, default=-1)
        return parent_parser

    def cleanup(self):
        self.sparse_voxelizer.cleanup()

