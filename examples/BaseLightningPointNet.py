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

from examples.utils import index_dict

class ChunkGeneratorCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # print("TRAIN GEN CHUNKS")
        trainer.datamodule.train_dataset.generate_chunks()
        
    def on_validation_epoch_start(self, trainer, pl_module):
        # print("VAL GEN CHUNKS")
        trainer.datamodule.val_dataset.generate_chunks()

class BasePointNetLightning(LightningDataModule):
    def __init__(self, **kwargs):
        # print("BasePointNetLightning init")
        super().__init__()
        # print(kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        self.seg_feat_channels = None
        # self.kwargs = copy.deepcopy(kwargs)
        # print(self.kwargs)
        self.labelweights=None
        # print("BasePointNetLightning init done")

    def setup(self, stage: Optional[str] = None):
        # print(self.kwargs)
        if self.use_whole_scene:
            self.train_dataset = self.whole_scene_dataset(phase="train", scene_list=self.train_files, **self.kwargs)
            self.val_dataset = self.whole_scene_dataset(phase="val", scene_list=self.val_files, **self.kwargs)
        else:
            self.train_dataset = self.chunked_scene_dataset(phase="train", scene_list=self.train_files, **self.kwargs)
            self.val_dataset = self.chunked_scene_dataset(phase="val", scene_list=self.val_files, **self.kwargs)
            # self.val_dataset = S3DISChunked(phase="val", scene_list=self.val_files, **self.kwargs)
        # val_kwargs = copy.deepcopy(self.kwargs)
        # val_kwargs['max_npoints'] = -1
        # val_kwargs['min_npoints'] = -1
        # self.val_dataset = self.whole_scene_dataset(phase="val", scene_list=self.val_files, **val_kwargs)
        # self.val_dataset.generate_chunks()
        
    
    def train_dataloader(self):
        # collate_fn = build_collate_fn(dense_input=self.dense_input, return_idx=self.return_point_idx)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      collate_fn=self.train_dataset.collate_fn, num_workers=self.num_workers,
                                      pin_memory=True, shuffle=True)
        # print(self.train_dataset)
        return train_dataloader

    def val_dataloader(self):
        # collate_fn = build_collate_fn(dense_input=self.dense_input, return_idx=self.return_point_idx)
        if self.val_split == 'train':
            val_dataset = self.train_dataset
        else:
            val_dataset = self.val_dataset
        # print(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                      collate_fn=val_dataset.collate_fn, num_workers=self.num_workers,
                                      pin_memory=True, shuffle=False)
        return val_dataloader

    def callbacks(self):
        if self.use_whole_scene:
            return []
        else:
            # print("Chunk Callback")
            return [ChunkGeneratorCallback()]

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("BasePointNetLightning")
        # parser.add_argument("--num_classes", type=int, default=20)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=6)
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--max_npoints", type=int, default=8192)
        parser.add_argument("--min_npoints", type=int, default=8192)
        
        # parser.add_argument("--voxel_size", type=float, default=.02)
        parser.add_argument("--use_whole_scene", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--return_point_idx", type=str2bool, nargs='?', const=True, default=False)

        parser.add_argument("--val_split", type=str, default='val')
        return parent_parser



class BasePointNet():
    def __init__(self, **kwargs):
        # print("BasePointNet init")
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        assert self.phase in ["train", "val", "test"]
        # print("BasePointNet init done")

    def collate_fn(self, data):
        batch_size = len(data)
        # batch_idx = torch.cat([torch.ones(data[i]['pts'].shape[0], 1) * i for i in range(batch_size)], axis=0)
        out_dict = {}
        for batch_idx, batch in enumerate(data):
            batch['batch_idx'] = batch_idx
            batch = self.process_input(batch, training=self.phase == 'train')
            for k, v in batch.items():
                if batch_idx == 0:
                    out_dict[k] = [v]
                else:
                    out_dict[k].append(v)

        for k, v in out_dict.items():
            if np.all([isinstance(it, torch.Tensor) for it in v]):
                if self.dense_input and k != 'labels':
                    # print(k, v[0].shape)
                    # print(v.)
                    out_dict[k] = torch.stack(v, axis=0)
                    # print(out_dict[k].shape)
                    if k == 'feats':
                        out_dict[k] = out_dict[k].transpose(1,2).contiguous()
                else:
                    out_dict[k] = torch.cat(v, axis=0)
                    # print(k, out_dict[k].shape)
            elif k == 'batch_idx' or k == 'num_pts':
                out_dict[k] = torch.tensor(v)
                
        # print(out_dict)
        # for k, v in 
        # assert(True == False)
        return out_dict

class BaseWholeScene(BasePointNet):
    def __init__(self, **kwargs):
        # print("BaseWholeScene init")
        super().__init__(**kwargs)
        # print(kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        assert self.phase in ["train", "val", "test"]

        self._load_scene_file()
        # print(self.scene_pillar_list, len(self.scene_pillar_list))
        # print("BaseWholeScene init done")

    def _load_scene_file(self):
        self.scene_pillar_list = []

        for scene_id in tqdm(self.scene_list):
            scene_data = self.load_sample(scene_id)

            coordmax = scene_data['pts'].max(axis=0)[0]
            coordmin = scene_data['pts'].min(axis=0)[0]
            xlength = 1.5
            ylength = 1.5
            nsubvolume_x = torch.ceil((coordmax[0]-coordmin[0])/xlength).int()
            nsubvolume_y = torch.ceil((coordmax[1]-coordmin[1])/ylength).int()
            # print(nsubvolume_x)
            for i in range(nsubvolume_x):
                for j in range(nsubvolume_y):
                    curmin = coordmin+torch.tensor([i*xlength, j*ylength, 0])
                    curmax = coordmin+torch.tensor([(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]])
                    mask = torch.sum((scene_data['pts']>=(curmin))*(scene_data['pts']<=(curmax)), axis=1)==3
                    cur_pillar_dict = index_dict(scene_data, mask)
                    
                    cur_seg = cur_pillar_dict['labels'].int()
                    if cur_seg.shape[0] == 0 or torch.all(cur_seg == -1):
                        continue

                    if self.return_point_idx:
                        cur_pillar_dict['point_idx'] = torch.arange(scene_data.shape[0])[mask]
                    
                    cur_num_pts = len(cur_seg)
                    if cur_num_pts > self.max_npoints and self.max_npoints > 0:
                        choice = torch.randperm(cur_num_pts)[:self.max_npoints]
                    elif cur_num_pts < self.min_npoints and self.min_npoints > 0:
                        p = torch.ones(cur_num_pts) / cur_num_pts
                        choice = p.multinomial(num_samples=self.min_npoints - cur_num_pts, replacement=True)
                        choice = torch.cat([torch.arange(cur_num_pts), choice])
                    else:
                        choice = torch.arange(cur_num_pts)

                    cur_pillar_dict = index_dict(cur_pillar_dict, choice)
                    cur_pillar_dict['pts'][:,:2] -= (curmin[:2] + curmax[:2])/2
                    cur_pillar_dict['pts'][:,2] -= cur_pillar_dict['pts'][:,2].min()
                    #assert(not np.all(cur_point_set[:,-1] == 156))
                    # assert(not torch.all(cur_point_set[:,-1] == -1))
                    # cur_pillar_dict['pts'] -= cur_pillar_dict['pts'][:,:2]
                    # assert(cur_pillar_dict['pts'].shape[0] != 0 and not torch.all(cur_pillar_dict['labels'] == -1))
                    self.scene_pillar_list.append(cur_pillar_dict)

    def __getitem__(self, index):
        start = time.time()
        in_dict = copy.deepcopy(self.scene_pillar_list[index])
        # in_dict = self.process_input(in_dict)
        fetch_time = time.time() - start
        return in_dict

    def __len__(self):
        return len(self.scene_pillar_list)

class BaseChunked(BasePointNet):
    def __init__(self, **kwargs):
        # print("BaseChunked init")
        super().__init__(**kwargs)
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        assert self.phase in ["train", "val", "test"]
        self.chunk_data = {} # init in generate_chunks()
        # print("BaseChunked init done")

    # @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        in_dict = copy.deepcopy(self.chunk_data[scene_id])
        # print(in_dict['pts'].shape)
        # print(in_dict)
        fetch_time = time.time() - start

        return in_dict

    def generate_chunks(self):
        """
            note: must be called before training
        """
        # print("generate new chunks for {}...".format(self.phase))
        # for scene_id in tqdm(self.scene_list):
        for scene_id in tqdm(self.scene_list):
            scene_data = self.load_sample(scene_id)

            coordmax = scene_data['pts'].max(axis=0)[0]
            coordmin = scene_data['pts'].min(axis=0)[0]
            semantic = scene_data['labels'].int()
            
            for _ in range(100):
                # curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curcenter = scene_data['pts'][torch.randint(len(semantic), (1,))][0]
                half_pillar_dims = torch.tensor([0.75,0.75,1.5])
                curmin = curcenter-half_pillar_dims
                curmax = curcenter+half_pillar_dims
                # print(curmin, curmax, curcenter)
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = torch.sum((scene_data['pts']>=(curmin))*(scene_data['pts']<=(curmax)),axis=1)==3
                cur_chunk_dict = index_dict(scene_data, curchoice)

                cur_num_pts = len(cur_chunk_dict['labels'])

                if cur_num_pts==0:
                    continue

                # print(mask, mask.shape)
                vidx = torch.ceil((cur_chunk_dict['pts']-curmin)/(curmax-curmin)*torch.tensor([31.0,31.0,62.0]))
                vidx = torch.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                # print(vidx)
                isvalid = torch.sum(cur_chunk_dict['labels']>=0)/cur_num_pts>=0.5 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            # cur_num_pts = len(cur_point_set)
            if cur_num_pts > self.max_npoints and self.max_npoints > 0:
                choice = torch.randperm(cur_num_pts)[:self.max_npoints]
            elif cur_num_pts < self.min_npoints and self.min_npoints > 0:
                p = torch.ones(cur_num_pts) / cur_num_pts
                choice = p.multinomial(num_samples=self.min_npoints - cur_num_pts, replacement=True)
                choice = torch.cat([torch.arange(cur_num_pts), choice])
            else:
                choice = torch.arange(cur_num_pts)
            # print(choice.shape)
            cur_chunk_dict = index_dict(cur_chunk_dict, choice)
            cur_chunk_dict['pts'][:,:2] -= curcenter[:2]
            cur_chunk_dict['pts'][:,2] -= cur_chunk_dict['pts'][:,2].min()
            self.chunk_data[scene_id] = cur_chunk_dict

    def __len__(self):
        return len(self.scene_list)