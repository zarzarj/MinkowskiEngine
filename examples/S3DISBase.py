import os
import glob
from typing import Any, Optional, List, NamedTuple

import torch
import numpy as np
from tqdm import tqdm
from examples.BaseDataset import BaseDataset
from examples.str2bool import str2bool

class S3DISBase(BaseDataset):
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        self.feat_channels = 3 * int(self.use_colors) 
        self.train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
        self.val_areas = ['Area_5']
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}
        self.class_labels = self.label_to_names.values()
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.NUM_LABELS = len(self.class_labels)
        if self.preprocess_max_num_pts < 0:
            self.preprocess_path = os.path.join(self.data_dir, 'preprocessing', 'base')
        else:
            self.preprocess_path = os.path.join(self.data_dir, 'preprocessing', 'num_pts_' + str(self.preprocess_max_num_pts))

    def prepare_data(self):
        os.makedirs(self.preprocess_path, exist_ok=True)
        train_rooms = [glob.glob(os.path.join(self.data_dir, area, '*[!.txt]')) for area in self.train_areas]
        train_rooms = [room for area_rooms in train_rooms for room in area_rooms]
        val_rooms = [glob.glob(os.path.join(self.data_dir, area, '*[!.txt]')) for area in self.val_areas]
        val_rooms = [room for area_rooms in val_rooms for room in area_rooms]
        # print(val_rooms)
        all_rooms = train_rooms + val_rooms
        # print(all_rooms)
        for room_folder in tqdm(all_rooms):
            room_name = room_folder.split('/')[-1]
            area_name = room_folder.split('/')[-2]
            room_files = glob.glob(os.path.join(self.preprocess_path, area_name + '_' + room_name + '*.pt'))
            if len(room_files) == 0:
                room = torch.from_numpy(self.load_room(room_folder))
                if self.preprocess_max_num_pts > 0 and room.shape[0] > self.preprocess_max_num_pts and room_folder in train_rooms:
                    room_file = os.path.join(self.preprocess_path, area_name + '_' + room_name)
                    self.partition_room(room, room_file)
                else:
                    room_file = os.path.join(self.preprocess_path, area_name + '_' + room_name + '_0.pt')
                    torch.save(room, room_file)

    def partition_room(self, room, room_file):
        pts = room[:,:3].numpy()
        min_coords = pts.min(axis=0)[:2]
        max_coords = pts.max(axis=0)[:2]
        side_len = max_coords - min_coords
        split_idx = int(side_len[0] <= side_len[1])
        for i in range(2): 
            cur_min = min_coords.copy()
            cur_min[split_idx] += i * side_len[split_idx] / 2
            cur_max = max_coords.copy()
            cur_max[split_idx] -= (1 - i) * side_len[split_idx] / 2
            # print(i, side_len, split_idx, cur_max, cur_min, max_coords, min_coords)
            # assert(True == False)
            mask = np.sum((pts[:,:2]>=(cur_min))*(pts[:,:2]<=(cur_max)),axis=1)==2
            room_file = room_file + '_' + str(i)
            masked_room = room[mask]
            if masked_room.shape[0] < self.preprocess_max_num_pts:
                # print(masked_room.shape)
                torch.save(masked_room, room_file + '.pt')
            else:
                self.partition_room(masked_room, room_file)

    def setup(self, stage: Optional[str] = None):
        self.train_files = [glob.glob(os.path.join(self.preprocess_path, area + '*')) for area in self.train_areas]
        self.train_files = [room for area_rooms in self.train_files for room in area_rooms]
        self.val_files = [glob.glob(os.path.join(self.preprocess_path, area + '*')) for area in self.val_areas]
        self.val_files = [room for area_rooms in self.val_files for room in area_rooms] 


    def load_room(self, room_folder):
        cloud_points = np.empty((0, 3), dtype=np.float32)
        cloud_colors = np.empty((0, 3), dtype=np.float32)
        cloud_classes = np.empty((0, 1), dtype=np.int32)
        # room_folder = osp.join(self.data_dir, area, room)
        for object_name in os.listdir(os.path.join(room_folder, 'Annotations')):
            if object_name[-4:] == '.txt':
                object_file = os.path.join(room_folder, 'Annotations', object_name)
                tmp = object_name[:-4].split('_')[0]
                if tmp in self.name_to_label:
                    object_class = self.name_to_label[tmp]
                elif tmp in ['stairs']:
                    object_class = self.name_to_label['clutter']
                else:
                    raise ValueError('Unknown object name: ' + str(tmp))
                with open(object_file, 'r') as f:
                    object_data = []
                    for i, line in enumerate(f):
                        object_data.append([float(x) for x in line.split()])
                    object_data = np.array(object_data)
                cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                cloud_classes = np.vstack((cloud_classes, object_classes))
        room = np.concatenate([cloud_points, cloud_colors, cloud_classes], axis=-1)
        return room

    def get_features(self, in_dict):
        feats = []
        if self.use_colors:
            feats.append(in_dict['colors'])
        out_feats = torch.cat(feats, dim=-1).float()
        if self.random_feats:
            out_feats = torch.rand_like(out_feats) - 0.5

        return out_feats

    def load_sample(self, idx):
        room_data = torch.load(idx)
        out_dict = {'pts': room_data[:, :3],  # include xyz by default
                    'labels': room_data[:, -1],
                    'scene_name': idx.split('/')[-1],
                    } 

        if self.use_colors:
            out_dict['colors'] = room_data[:, 3:6]

        if self.load_graph:
            out_dict['adj'] = torch.load(os.path.join('/',*idx.split('/')[:-1], 'adjs', idx.split('/')[-1] + '_adj.pt'))

        return out_dict

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseDataset.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("S3DISBase")
        parser.add_argument("--preprocess_max_num_pts", type=int, default=500000)
        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--load_graph", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("--use_augmentation", type=str2bool, nargs='?', const=True, default=True)
        return parent_parser