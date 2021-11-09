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
from examples.BaseLightningPointNet import BasePointNetLightning, BaseWholeScene, BaseChunked
# from prefetch_generator import background

class ChunkGeneratorCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.train_dataset.generate_chunks()
        
    def on_validation_epoch_start(self, trainer, pl_module):
        trainer.datamodule.val_dataset.generate_chunks()

class ScanNetPointNet(BasePointNetLightning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
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

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or self.val_split == 'train':
            with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
                self.train_files = f.readlines()
                self.train_files = [file[:-5] for file in self.train_files]
            if self.use_whole_scene:
                self.train_dataset = ScannetWholeScene(phase="train", scene_list=self.train_files,
                                                              label_map=self.label_map, labelweights=self.labelweights,
                                                              **self.kwargs)
            else:
                self.train_dataset = ScannetChunked(phase="train", scene_list=self.train_files, label_map=self.label_map,
                                                    labelweights=self.labelweights, **self.kwargs)
            # self.labelweights = self.train_dataset.labelweights
            # self.train_dataset.generate_chunks()
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
            self.val_files = f.readlines()
            self.val_files = [file[:-5] for file in self.val_files]
        if self.use_whole_scene:
            self.val_dataset = ScannetWholeScene(phase="val", scene_list=self.val_files, label_map=self.label_map,
                                                        labelweights=self.labelweights, **self.kwargs)
        else:
            self.val_dataset = ScannetChunked(phase="val", scene_list=self.val_files, label_map=self.label_map,
                                              labelweights=self.labelweights, **self.kwargs)
        # self.val_dataset.generate_chunks()

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePointNetLightning.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetPointNetLightning")
        return parent_parser


class ScannetWholeScene(BaseWholeScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        # assert self.phase in ["train", "val", "test"]

        # self._load_scene_file()

    def load_scene(self, scene_id):
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
                scene_data[scene_data[:,-1] >= 156,9] = -1
        else:
            scene_data = np.load(os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', scene_id + '.npy'))
        return scene_data


class ScannetChunked(BaseChunked):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        # assert self.phase in ["train", "val", "test"]
        # self.chunk_data = {} # init in generate_chunks()
        # self._prepare_weights()

    def load_scene(self, scene_id):
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
                scene_data[scene_data[:,-1] >= 156,9] = -1
        else:
            scene_data = np.load(os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', scene_id + '.npy'))
        return scene_data

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