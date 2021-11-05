import os
import glob
import time
import math
import inspect
from typing import Any, Optional, List, NamedTuple

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import numpy as np
from tqdm import tqdm
from plyfile import PlyElement, PlyData

# import MinkowskiEngine as ME

# from examples.voxelizer import SparseVoxelizer
# import examples.transforms as t
from examples.str2bool import str2bool
from examples.ScanNetLightning import ScanNet
import torch_geometric
from examples.utils import sparse_collate
# from examples.utils import get_embedder

def save_adj(PC, adj, filename):
    from plyfile import PlyElement, PlyData
    PC = [tuple(element) for element in PC]
    vertex_el = PlyElement.describe(np.array(PC, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
    faces = np.zeros((len(PC), 3), dtype=np.int32)
    for i in range(len(PC)):
        faces[i, :] = adj[0,i*16:i*16+3]
    faces = [(element, 0, 0, 0) for element in faces]
    # faces[0] = (faces[0][0], 255, 0, 0)
    # for i in range(20):
    #     faces[i] = (faces[i][0], 255, 0, 0)
    # print(faces[0])
    face_el = PlyElement.describe(np.array(faces, dtype=[ ('vertex_indices', 'i4', (3,)),
                                                          ('red', 'u1'), ('green', 'u1'),
                                                          ('blue', 'u1')
                                                        ]), 'face')
    PlyData([vertex_el, face_el]).write(filename)

class ScanNetLIG_graph(ScanNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(self):
        all_scans = glob.glob(os.path.join(self.scans_dir, '*')) + glob.glob(os.path.join(self.scans_test_dir, '*'))
        os.makedirs(os.path.join(self.data_dir, 'adjs'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'subsample_idx'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'num_pts'), exist_ok=True)
        num_pts_file = os.path.join(self.data_dir, 'num_pts.pt')
        if not os.path.exists(num_pts_file):
            num_pts = {}
            for scan in all_scans:
                scene_name = scan.split('/')[-1]
                scan_file = os.path.join(scan, scene_name + '_vh_clean_2.ply')
                with open(scan_file, 'rb') as f:
                    plydata = PlyData.read(f)
                # pts = torch.from_numpy(np.stack((plydata['vertex']['x'],
                #                        plydata['vertex']['y'],
                #                        plydata['vertex']['z'])).T)
                num_pts[scene_name] = plydata['vertex']['x'].shape[0]
                # print(plydata['vertex']['x'].shape[0])
            torch.save(num_pts, num_pts_file)
        else:
            num_pts = torch.load(num_pts_file)

        for scan in all_scans:
            scene_name = scan.split('/')[-1]
            if self.max_num_pts > 0 and self.max_num_pts < num_pts[scene_name]:
                adj_file = os.path.join(self.data_dir, 'adjs', scene_name + f'_adj_{self.max_num_pts}.pt')
            else:
                adj_file = os.path.join(self.data_dir, 'adjs', scene_name + '_adj.pt')
            if not os.path.exists(adj_file):
                # print(pts.shape)
                scan_file = os.path.join(scan, scene_name + '_vh_clean_2.ply')
                with open(scan_file, 'rb') as f:
                    plydata = PlyData.read(f)
                pts = torch.from_numpy(np.stack((plydata['vertex']['x'],
                                       plydata['vertex']['y'],
                                       plydata['vertex']['z'])).T)
                if self.max_num_pts > 0 and self.max_num_pts < pts.shape[0]:
                    subsample_idx_file = os.path.join(self.data_dir, 'subsample_idx', scene_name + f'_{self.max_num_pts}_idx.pt')
                    if not os.path.exists(subsample_idx_file):
                        subsample_idx = torch.randperm(pts.shape[0])[:self.max_num_pts]
                        torch.save(subsample_idx, subsample_idx_file)
                    else:
                        subsample_idx = torch.load(subsample_idx_file)
                    pts = pts[subsample_idx]

                adj = torch_geometric.nn.pool.knn_graph(x=pts, k=16,
                                                    loop=False, flow='source_to_target',
                                                    cosine=False)
                torch.save(adj, adj_file)

        
    def load_ply(self, idx):
        in_dict = super().load_ply(idx)
        if self.max_num_pts > 0 and self.max_num_pts < in_dict['pts'].shape[0]:
            adj_file = os.path.join(self.data_dir, 'adjs', idx + f'_adj_{self.max_num_pts}.pt')
        else:
            adj_file = os.path.join(self.data_dir, 'adjs', idx + '_adj.pt')
        in_dict['adjacency'] = torch.load(adj_file)
        return in_dict

    def process_input(self, input_dict):
        if self.trainer.training and self.max_num_pts > 0 and self.max_num_pts < input_dict['pts'].shape[0]:
            subsample_idx_file = os.path.join(self.data_dir, 'subsample_idx', input_dict['scene_name'] + f'_{self.max_num_pts}_idx.pt')
            perm = torch.load(subsample_idx_file)
        else:
            perm = torch.arange(input_dict['pts'].shape[0])

        input_dict['pts'] = input_dict['pts'][perm]
        input_dict['colors'] = input_dict['colors'][perm]
        input_dict['labels'] = input_dict['labels'][perm]
        if self.use_implicit_feats:
            input_dict['implicit_feats'] = input_dict['implicit_feats'][perm]
        input_dict['colors'] = (input_dict['colors'] / 255.) - 0.5
        input_dict['feats'] = self.get_features(input_dict)
        input_dict['coords'] = input_dict['pts'] / self.voxel_size
        input_dict['seg_feats'] = None
        input_dict['rand_shift'] = None
        return input_dict

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        coords_batch, feats_batch, labels_batch = sparse_collate(input_dict['coords'], input_dict['feats'], input_dict['labels'],
                                                                          dtype=torch.float32)
        merged_adj = collate_adjs(input_dict['adjacency'])
        update_dict = {"coords": coords_batch,
                    "labels": labels_batch.long(),
                    "feats": feats_batch,
                    "idxs": idxs,
                    "adj": merged_adj}
        input_dict.update(update_dict)
        return input_dict

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = ScanNet.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetLIG_graph")
        return parent_parser

def collate_adjs(adjs):
    num_pts = 0
    # print(adjs)
    for i in range(len(adjs)):
        # print(adjs[i])
        adjs[i] += num_pts
        num_pts += adjs[i].shape[0]
        # print(adjs[i].shape)
    return torch.cat(adjs, axis=1)