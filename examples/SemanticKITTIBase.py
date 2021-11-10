import os
import glob
from typing import Any, Optional, List, NamedTuple

import torch
import numpy as np
from tqdm import tqdm

class SemanticKITTIBase():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)

        self.train_seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        self.val_seqs = ['08']
        self.test_seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.label_to_names = {0 : "unlabeled",
                              1 : "outlier",
                              10: "car",
                              11: "bicycle",
                              13: "bus",
                              15: "motorcycle",
                              16: "on-rails",
                              18: "truck",
                              20: "other-vehicle",
                              30: "person",
                              31: "bicyclist",
                              32: "motorcyclist",
                              40: "road",
                              44: "parking",
                              48: "sidewalk",
                              49: "other-ground",
                              50: "building",
                              51: "fence",
                              52: "other-structure",
                              60: "lane-marking",
                              70: "vegetation",
                              71: "trunk",
                              72: "terrain",
                              80: "pole",
                              81: "traffic-sign",
                              99: "other-object",
                              252: "moving-car",
                              253: "moving-bicyclist",
                              254: "moving-person",
                              255: "moving-motorcyclist",
                              256: "moving-on-rails",
                              257: "moving-bus",
                              258: "moving-truck",
                              259: "moving-other-vehicle",}
        self.label_map = {
                          0 : 0,     # "unlabeled"
                          1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                          10: 1,     # "car"
                          11: 2,     # "bicycle"
                          13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
                          15: 3,     # "motorcycle"
                          16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
                          18: 4,     # "truck"
                          20: 5,     # "other-vehicle"
                          30: 6,     # "person"
                          31: 7,     # "bicyclist"
                          32: 8,     # "motorcyclist"
                          40: 9,     # "road"
                          44: 10,    # "parking"
                          48: 11,    # "sidewalk"
                          49: 12,    # "other-ground"
                          50: 13,    # "building"
                          51: 14,    # "fence"
                          52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
                          60: 9,     # "lane-marking" to "road" ---------------------------------mapped
                          70: 15,    # "vegetation"
                          71: 16,    # "trunk"
                          72: 17,    # "terrain"
                          80: 18,    # "pole"
                          81: 19,    # "traffic-sign"
                          99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
                          252: 1,    # "moving-car" to "car" ------------------------------------mapped
                          253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
                          254: 6,    # "moving-person" to "person" ------------------------------mapped
                          255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
                          256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
                          257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
                          258: 4,    # "moving-truck" to "truck" --------------------------------mapped
                          259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
                          }
        self.class_labels = self.label_to_names.values()
        self.NUM_LABELS = len(set(self.label_map.values()))

    def prepare_data(self):
        preprocess_path = os.path.join(self.data_dir, 'preprocessed')
        os.makedirs(preprocess_path, exist_ok=True)
        train_files = [glob.glob(os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'velodyne', '*.bin')) for seq in self.train_seqs]
        train_files = [room for area_rooms in train_files for room in area_rooms]
        val_files = [glob.glob(os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'velodyne', '*.bin')) for seq in self.val_seqs]
        val_files = [room for area_rooms in val_files for room in area_rooms]
        scan_splits = {'train': train_scans, 'val': val_files}
        for split, scans in scan_splits.items():
            os.makedirs(os.path.join(preprocess_path, split), exist_ok=True)
            for i, scan in tqdm(enumerate(scans)):
                unique_scene_id = f'{i:06d}'
                processed_scan_file = os.path.join(preprocess_path, split, unique_scene_id + '.pt')
                if not os.path.exists(processed_scan_file):
                    pts = np.fromfile(scan, dtype=np.float32).reshape(-1,4)
                    seq = scan.split('/')[-3]
                    scan_id = scan.split('/')[-1].split('.')[0]
                    label_file = os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'labels', scan_id + '.label')
                    labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF
                    labels = np.array([self.label_map[label] - 1 for label in labels]).reshape(-1,1)
                    pts = torch.from_numpy(np.concatenate([pts, labels], axis=1))
                    torch.save(pts, processed_scan_file)

    def setup(self, stage: Optional[str] = None):
        preprocess_path = os.path.join(self.data_dir, 'preprocessed')
        self.train_files = glob.glob(os.path.join(self.preprocess_path, 'train', '*'))
        self.val_files = glob.glob(os.path.join(self.preprocess_path, 'val', '*'))

    def load_sample(self, idx):
        room_data = torch.load(idx)
        out_dict = {'pts': room_data[:, :3],  # include xyz by default
                    'labels': room_data[:, -1],
                    'scene_name': idx.split('/')[-1].split('.')[0],
                    } 
        return out_dict

    def load_scene(self, idx):
        return torch.load(idx)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SemanticKITTIBase")
        return parent_parser
