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
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
            self.train_files = f.readlines()
            self.train_files = [file[:-5] for file in self.train_files]
        self.train_dataset = ScannetDataset(phase="train", scene_list=self.train_files, **self.kwargs)
        self.labelweights = torch.from_numpy(self.train_dataset.labelweights)

    def setup(self, stage: Optional[str] = None):
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
            self.train_files = f.readlines()
            self.train_files = [file[:-5] for file in self.train_files]
        self.train_dataset = ScannetDataset(phase="train", scene_list=self.train_files, **self.kwargs)
        # self.labelweights = self.train_dataset.labelweights
        # self.train_dataset.generate_chunks()
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
            self.val_files = f.readlines()
            self.val_files = [file[:-5] for file in self.val_files]
        self.val_dataset = ScannetDataset(phase="val", scene_list=self.val_files, **self.kwargs)
        # self.val_dataset.generate_chunks()
        
    
    def train_dataloader(self):
        if self.dense_input:
            collate_fn=collate_random
        else:
            collate_fn=collate_random_PyG
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      collate_fn=collate_fn, num_workers=self.num_workers,
                                      pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        if self.dense_input:
            collate_fn=collate_random
        else:
            collate_fn=collate_random_PyG
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                                      collate_fn=collate_fn, num_workers=self.num_workers,
                                      pin_memory=True)
        return val_dataloader

    def callbacks(self):
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
        parser.add_argument("--npoints", type=int, default=8192)

        parser.add_argument("--use_implicit", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_color", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_normal", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--random_feats", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--weighting", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--augment_points", type=str2bool, nargs='?', const=True, default=True)

        parser.add_argument("--overlap_factor", type=int, default=2)
        parser.add_argument("--voxel_size", type=float, default=.25)

        parser.add_argument("--dense_input", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser


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

        fetch_time = time.time() - start

        if self.random_feats:
            point_set[:,3:] = np.random.rand(point_set.shape[0], point_set.shape[1]-3)

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

            choices = np.random.choice(cur_point_set.shape[0], self.npoints, replace=True)
            cur_point_set = cur_point_set[choices]
            self.chunk_data[scene_id] = cur_point_set
            
        # print("done!\n")

def collate_random(data):
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
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # split points to coords and feats
    coords = point_set[:, :, :3] #B, N, 3
    feats = point_set[:, :, 3:].contiguous().transpose(1,2) #B, C, N

    # pack
    batch = {'coords': coords,
             'feats': feats,
             'labels': [sem for sem in semantic_seg],
    }
    return batch

def collate_random_PyG(data):
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
    (   point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # print(semantic_seg, semantic_seg.min(), semantic_seg.max())

    # split points to coords and feats
    batch_size = point_set.shape[0]
    num_pts = point_set.shape[1]
    batch_idx = torch.arange(batch_size).repeat_interleave(num_pts).unsqueeze(1)
    # print(batch_idx, batch_idx.shape)
    coords = point_set[:, :, :3]
    coords = coords.contiguous().reshape(-1, coords.shape[-1])
    coords = torch.cat([batch_idx, coords], axis=1)

    feats = point_set[:, :, 3:]
    feats = feats.contiguous().reshape(-1, feats.shape[-1])

    # pack
    batch = {'coords': coords,
             'feats': feats,
             'labels': [sem for sem in semantic_seg],
    }

    return batch