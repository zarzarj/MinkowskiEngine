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

import multiprocessing as mp
from tqdm import tqdm
from pytorch_lightning.callbacks import Callback
from plyfile import PlyData, PlyElement
# from prefetch_generator import background

class ChunkGeneratorCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.train_dataset.generate_chunks()
        
    def on_validation_epoch_start(self, trainer, pl_module):
        trainer.datamodule.val_dataset.generate_chunks()

class ScanNetPointNet(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)  
        self.feat_channels = 32 * int(self.use_implicit) \
                           + 3 * int(self.use_color) + 3 * int(self.use_normal)
        self.seg_feat_channels = None
        self.kwargs = kwargs
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
                self.label_map[l] = -1
            else:
                self.label_map[l] = n_used
                n_used += 1
        self.label_map[-1] = -1
        self.NUM_LABELS -= len(self.IGNORE_LABELS)

        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
            self.train_files = f.readlines()
            self.train_files = [file[:-5] for file in self.train_files]
        # self.train_dataset = ScannetDataset(phase="train", scene_list=self.train_files, **self.kwargs)
        # self.labelweights = torch.from_numpy(self.train_dataset.labelweights)
        self.labelweights=None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or self.val_split == 'train':
            with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
                self.train_files = f.readlines()
                self.train_files = [file[:-5] for file in self.train_files]
            if self.use_whole_scene:
                self.train_dataset = ScannetDatasetWholeScene(phase="train", scene_list=self.train_files,
                                                              label_map=self.label_map, labelweights=self.labelweights,
                                                              **self.kwargs)
            else:
                self.train_dataset = ScannetDataset(phase="train", scene_list=self.train_files, label_map=self.label_map,
                                                    labelweights=self.labelweights, **self.kwargs)
            # self.labelweights = self.train_dataset.labelweights
            # self.train_dataset.generate_chunks()
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
            self.val_files = f.readlines()
            self.val_files = [file[:-5] for file in self.val_files]
        if self.use_whole_scene:
            self.val_dataset = ScannetDatasetWholeScene(phase="val", scene_list=self.val_files, label_map=self.label_map,
                                                        labelweights=self.labelweights, **self.kwargs)
        else:
            self.val_dataset = ScannetDataset(phase="val", scene_list=self.val_files, label_map=self.label_map,
                                              labelweights=self.labelweights, **self.kwargs)
        # self.val_dataset.generate_chunks()
        
    
    def train_dataloader(self):
        collate_fn = build_collate_fn(dense_input=self.dense_input, return_idx=self.return_point_idx)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      collate_fn=collate_fn, num_workers=self.num_workers,
                                      pin_memory=True, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        collate_fn = build_collate_fn(dense_input=self.dense_input, return_idx=self.return_point_idx)
        if self.val_split == 'train':
            val_dataset = self.train_dataset
        else:
            val_dataset = self.val_dataset
            
        val_dataloader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                      collate_fn=collate_fn, num_workers=self.num_workers,
                                      pin_memory=True, shuffle=False)
        return val_dataloader

    def callbacks(self):
        if self.use_whole_scene:
            return []
        else:
            return [ChunkGeneratorCallback()]

    # def on_train_epoch_start(self):
    #     print("TRAIN EPOCH START")
    #     self.train_dataset.generate_chunks()

    # def on_validation_epoch_start(self):
    #     print("VAL EPOCH START")
    #     self.val_dataset.generate_chunks()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ScanNetPointNet")
        parser.add_argument("--num_classes", type=int, default=20)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=6)
        parser.add_argument("--test_batch_size", type=int, default=6)
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--max_npoints", type=int, default=8192)
        parser.add_argument("--min_npoints", type=int, default=8192)

        parser.add_argument("--use_implicit", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_color", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_normal", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--random_feats", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--weighting", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--augment_points", type=str2bool, nargs='?', const=True, default=True)

        parser.add_argument("--overlap_factor", type=int, default=2)
        parser.add_argument("--voxel_size", type=float, default=.25)

        parser.add_argument("--dense_input", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_whole_scene", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--return_point_idx", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_orig_pcs", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--val_split", type=str, default='val')
        return parent_parser


class ScannetDatasetWholeScene():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        assert self.phase in ["train", "val", "test"]

        self._load_scene_file()

    def _load_scene_file(self):
        self.scene_points_list = []
        if self.return_point_idx:
            self.point_idx_list = []
            self.scene_id_list = []
        if self.use_implicit:
            self.implicit_data = []

        for scene_id in tqdm(self.scene_list):
            if self.use_orig_pcs:
                scene_file = os.path.join(self.data_dir, 'scans_processed_full_normals', scene_id + '.npy')
                if not os.path.exists(scene_file):
                    scan_file = os.path.join(self.data_dir, 'scans', scene_id, scene_id + '_vh_clean_2.ply')
                    with open(scan_file, 'rb') as f:
                        plydata = PlyData.read(f)
                    # print(plydata['vertex'])
                    pts = np.stack((plydata['vertex']['x'],
                                           plydata['vertex']['y'],
                                           plydata['vertex']['z'])).T
                    colors = np.stack((plydata['vertex']['red'],
                               plydata['vertex']['green'],
                               plydata['vertex']['blue'])).T
                    xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in plydata["vertex"].data])
                    face = np.array([f[0] for f in plydata["face"].data])
                    normals = compute_normal(xyz, face)
                    # print(normals.shape)
                    label_file = scan_file[:-4] + '.labels.ply'
                    with open(label_file, 'rb') as f:
                        plydata = PlyData.read(f)
                    labels = np.array(plydata['vertex']['label'], dtype=np.uint8)
                    labels = np.array([self.label_map[x] for x in labels], dtype=np.uint8)
                    scene_data = np.concatenate([pts, colors, normals, labels.reshape(-1,1)], axis=-1)
                    np.save(scene_file, scene_data)
                else:
                    scene_data = np.load(scene_file)
                    scene_data[scene_data[:,9] >= 156,9] = -1
                    # print(scene_data[:,9].max())
                    # print(scene_data.shape)
            else:
                scene_data = np.load(os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', scene_id + '.npy'))
            if self.use_implicit:
                implicit_feats = np.load(os.path.join(self.data_dir, 'implicit_feats_pointnet2', scene_id + '.feats.npy'))
                scene_data = np.concatenate([scene_data, implicit_feats], axis=-1)

            coordmax = scene_data[:, :3].max(axis=0)
            coordmin = scene_data[:, :3].min(axis=0)
            xlength = 1.0
            ylength = 1.0
            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)
            for i in range(nsubvolume_x):
                for j in range(nsubvolume_y):
                    curmin = coordmin+[i*xlength, j*ylength, 0]
                    curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                    mask = np.sum((scene_data[:, :3]>=(curmin))*(scene_data[:, :3]<=(curmax)), axis=1)==3
                    cur_point_set = scene_data[mask,:]
                    
                    # cur_semantic_seg = scene_data[mask, 10].astype(np.int32)
                    if self.use_orig_pcs:
                        cur_seg = cur_point_set[:,9].astype(np.int32)
                    else:
                        cur_seg = cur_point_set[:,10].astype(np.int32)
                    if cur_point_set.shape[0] == 0 or np.all(cur_seg == -1):
                        continue

                    if self.return_point_idx:
                        point_idx = np.arange(scene_data.shape[0])[mask]
                    
                    # print(len(cur_point_set))
                    cur_num_pts = len(cur_point_set)
                    if cur_num_pts > self.max_npoints and self.max_npoints > 0:
                        choice = np.random.choice(cur_num_pts, self.max_npoints, replace=False)
                    elif cur_num_pts < self.min_npoints and self.min_npoints > 0:
                        choice = np.random.choice(cur_num_pts, self.min_npoints - cur_num_pts, replace=True)
                        choice = np.concatenate([np.arange(cur_num_pts), choice])
                    else:
                        choice = np.arange(cur_num_pts)
                    cur_point_set = cur_point_set[choice,:] # Nx3
                    if self.return_point_idx:
                        point_idx = point_idx[choice]
                    #assert(not np.all(cur_point_set[:,-1] == 156))
                    assert(not np.all(cur_point_set[:,-1] == -1))


                        # mask = mask[choice]
                    self.scene_points_list.append(cur_point_set)
                    if self.return_point_idx:
                        self.point_idx_list.append(point_idx)
                        self.scene_id_list.append(scene_id)

        if self.weighting:
            labelweights = np.zeros(self.num_classes)
            for scene_data in self.scene_points_list:
                if self.use_orig_pcs:
                    seg = scene_data[:,9].astype(np.int32)
                else:
                    seg = scene_data[:,10].astype(np.int32)
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    def __getitem__(self, index):
        start = time.time()
        scene_data = self.scene_points_list[index]
        # mask = self.mask_list[index]

        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        color = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        if self.use_orig_pcs:
            label = scene_data[:,9].astype(np.int32)
        else:
            label = scene_data[:,10].astype(np.int32)

        if self.use_color:
            point_set = np.concatenate([point_set, color], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.use_implicit:
            implicit_features = scene_data[:, -32:]
            point_set = np.concatenate([point_set, implicit_features], axis=1)

        if self.phase == "train" and self.augment_points:
            point_set = self._augment(point_set)

        # mask = label == -100
        # sample_weight = self.labelweights[label]
        # sample_weight *= mask

        if self.random_feats:
            point_set[:,3:] = np.random.rand(point_set.shape[0], point_set.shape[1]-3)

        fetch_time = time.time() - start

        if self.return_point_idx:
            return point_set, label, None, fetch_time, self.point_idx_list[index], self.scene_id_list[index], index
        return point_set, label, None, fetch_time

    def __len__(self):
        return len(self.scene_points_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

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

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

class ScannetDataset():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        assert self.phase in ["train", "val", "test"]
        self.chunk_data = {} # init in generate_chunks()
        self._prepare_weights()

    def _prepare_weights(self):
        self.scene_data = {}
        scene_points_list = []
        semantic_labels_list = []
        if self.use_implicit:
            self.implicit_data = {}
        for scene_id in tqdm(self.scene_list):
            scene_file = os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', scene_id + '.npy')
            scene_data = np.load(scene_file)
            label = scene_data[:, 10]

            # append
            scene_points_list.append(scene_data)
            semantic_labels_list.append(label)
            self.scene_data[scene_id] = scene_data
            if self.use_implicit:
                self.implicit_data[scene_id] = np.load(os.path.join(self.data_dir, 'implicit_feats_pointnet2', scene_id + '.feats.npy'))

        if self.weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    # @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.chunk_data[scene_id]
        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        label = scene_data[:, 10].astype(np.int32)

        if self.use_implicit:
            implicit = scene_data[:, -32:]

        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.use_implicit:
            point_set = np.concatenate([point_set, implicit], axis=1)

        if self.phase == "train" and self.augment_points:
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask


        if self.random_feats:
            point_set[:,3:] = np.random.rand(point_set.shape[0], point_set.shape[1]-3)

        fetch_time = time.time() - start

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

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

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """
        # print("generate new chunks for {}...".format(self.phase))
        # for scene_id in tqdm(self.scene_list):
        for scene_id in self.scene_list:
            scene = self.scene_data[scene_id]
            semantic = scene[:, 10].astype(np.int32)

            coordmax = np.max(scene, axis=0)[:3]
            coordmin = np.min(scene, axis=0)[:3]
            
            for _ in range(5):
                curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene[curchoice]
                cur_semantic_seg = semantic[curchoice]

                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            if self.use_implicit:
                implicit_data = self.implicit_data[scene_id][curchoice]
                cur_point_set = np.concatenate([cur_point_set, implicit_data], axis=1)

            cur_num_pts = len(cur_point_set)
            if cur_num_pts > self.max_npoints and self.max_npoints > 0:
                choice = np.random.choice(cur_num_pts, self.max_npoints, replace=False)
            elif cur_num_pts < self.min_npoints and self.min_npoints > 0:
                choice = np.random.choice(cur_num_pts, self.min_npoints - cur_num_pts, replace=True)
                choice = np.concatenate([np.arange(cur_num_pts), choice])
            else:
                choice = np.arange(cur_num_pts)
            cur_point_set = cur_point_set[choice,:] # Nx3
            self.chunk_data[scene_id] = cur_point_set
            
        # print("done!\n")

def build_collate_fn(dense_input=False, return_idx=False):
    def collate_fn(data):
        '''
        for ScannetDataset: collate_fn=collate_random

        return: 
            coords               # torch.FloatTensor(B, N, 3)
            feats                # torch.FloatTensor(B, N, 3)
            semantic_segs        # torch.FloatTensor(B, N)
            sample_weights       # torch.FloatTensor(B, N)
            fetch_time           # float
        '''

        # load data
        if return_idx:
            (
                point_set, 
                semantic_seg, 
                sample_weight,
                fetch_time,
                point_idx,
                scene_id,
                index
            ) = zip(*data)
        else:
            (
                point_set, 
                semantic_seg, 
                sample_weight,
                fetch_time 
            ) = zip(*data)
            
        # print(point_set)
        if not dense_input or self.max_npoints < 0 or self.min_npoints < 0:
            batch_size = len(point_set)
            batch_idx = torch.cat([torch.ones(point_set[i].shape[0], 1) * i for i in range(batch_size)], axis=0)
            # batch_idx = torch.arange(batch_size)
            # coords = coords.contiguous().reshape(-1, coords.shape[-1])
            point_set = np.concatenate(point_set, axis=0)
            coords = torch.from_numpy(point_set[:, :3])
            coords = torch.cat([batch_idx, coords], axis=-1).float()
            feats = torch.from_numpy(point_set[:, 3:]).float()
            semantic_seg = torch.from_numpy(np.concatenate(semantic_seg, axis=0)).long()
            if return_idx:
                point_idx = torch.from_numpy(np.concatenate(point_idx, axis=0)).long()
        else:
            point_set = torch.FloatTensor(point_set)
            semantic_seg = torch.LongTensor(semantic_seg)
            if return_idx:
                point_idx = torch.LongTensor(point_idx)

            coords = point_set[:, :, :3]
            # feats = point_set[:, :, 3:]
            feats = point_set[:, :, 3:].contiguous().transpose(1,2) #B, C, N

        # pack
        batch = {'coords': coords,
                 'feats': feats,
                 'labels': semantic_seg.reshape(-1).long(),
        }
        if return_idx:
            batch['point_idx'] = point_idx
            batch['scene_id'] = scene_id
            batch['scene_index'] = index
        return batch
    return collate_fn


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)                
    return arr

def compute_normal(vertices, faces):
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]        
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    normalize_v3(n)
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    
    return normals