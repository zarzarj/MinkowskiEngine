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
        for scan in all_scans:
            scene_name = scan.split('/')[-1]
            adj_file = os.path.join(self.data_dir, 'adjs', scene_name + '_adj.pt')
            os.makedirs(os.path.join(self.data_dir, 'adjs'), exist_ok=True)
            if not os.path.exists(adj_file):
                scan_file = os.path.join(scan, scene_name + '_vh_clean_2.ply')
                with open(scan_file, 'rb') as f:
                    plydata = PlyData.read(f)
                pts = torch.from_numpy(np.stack((plydata['vertex']['x'],
                                       plydata['vertex']['y'],
                                       plydata['vertex']['z'])).T)
                # print(pts.shape)
                adj = torch_geometric.nn.pool.knn_graph(x=pts, k=16,
                                                    loop=False, flow='source_to_target',
                                                    cosine=False)
                # save_adj(pts, adj, 'test_adj.ply')
                # assert(True==False)
                # print(adj.shape)
                torch.save(adj, adj_file)

        
    def load_ply(self, idx):
        in_dict = super().load_ply(idx)
        adj_file = os.path.join(self.data_dir, 'adjs', idx + '_adj.pt')
        in_dict['adj'] = torch.load(adj_file)
        return in_dict

    def process_input(self, input_dict):
        input_dict['colors'] = (input_dict['colors'] / 255.) - 0.5
        input_dict['feats'] = self.get_features(input_dict)
        input_dict['coords'] = input_dict['pts'] / self.voxel_size
        input_dict['seg_feats'] = None
        input_dict['rand_shift'] = None
        return input_dict

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        coords_batch, feats_batch = sparse_collate(input_dict['coords'], input_dict['feats'],
                                                                          dtype=torch.float32)
        merged_adj = collate_adjs(input_dict['adj'])
        update_dict = {"coords": coords_batch,
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