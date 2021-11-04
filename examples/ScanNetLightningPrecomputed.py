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
import examples.transforms_dict as t
from examples.str2bool import str2bool
from examples.utils import interpolate_grid_feats, get_embedder, gather_nd, sparse_collate, sort_coords
from examples.BaseLightningPrecomputedDataset import BasePrecomputed
# from examples.DeepGCN_nopyg.gcn_lib.dense.torch_edge import dense_knn_matrix
# import MinkowskiEngine as ME


class ScanNetPrecomputed(BasePrecomputed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            -1: (255., 0., 0.),
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

    def prepare_data(self):
        if self.load_graph:
            if self.precompute_adjs:
                import MinkowskiEngine as ME
            import torch_geometric
            all_scans = glob.glob(os.path.join(self.scans_dir, '*')) + glob.glob(os.path.join(self.scans_test_dir, '*'))
            os.makedirs(os.path.join(self.data_dir, 'adjs'), exist_ok=True)
            for scan in tqdm(all_scans):
                scene_name = scan.split('/')[-1]
                adj_file = os.path.join(self.data_dir, 'adjs', scene_name + '_adj.pt')
                if not os.path.exists(adj_file):
                    scan_file = os.path.join(scan, scene_name + '_vh_clean_2.ply')
                    with open(scan_file, 'rb') as f:
                        plydata = PlyData.read(f)
                    pts = torch.from_numpy(np.stack((plydata['vertex']['x'],
                                           plydata['vertex']['y'],
                                           plydata['vertex']['z'])).T)
                    adj = torch_geometric.nn.pool.knn_graph(x=pts, k=16,
                                                    loop=False, flow='source_to_target',
                                                    cosine=False)
                    torch.save(adj, adj_file)
                    if self.precompute_adjs:
                        coords = torch.cat([torch.ones(pts.shape[0], 1), (pts / self.voxel_size).contiguous()], axis=-1)
                        in_field = ME.TensorField(
                            features=torch.ones_like(coords),
                            coordinates=coords,
                            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                            # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
                            # device=in_dict['feats'].device,
                        )

                        knn_file_base = os.path.join(self.data_dir, 'knns')
                        up_coords = pts
                        for i in range(4):
                            down_factor = 2**i
                            os.makedirs(os.path.join(knn_file_base, 'down_' + str(down_factor)), exist_ok=True)
                            down = in_field.sparse(down_factor)
                            print(down)
                            down_coords = down._C[:,1:].float()
                            down_coords = sort_coords(down_coords)
                            knn = torch_geometric.nn.pool.knn(up_coords, down_coords, k=16, num_workers=1)
                            torch.save(knn, os.path.join(knn_file_base, 'down_' + str(down_factor), scene_name + '_knn.pt'))
                            up_coords = down_coords
                        
                        # coords = torch.cat([torch.ones(pts.shape[0], 1), (pts / self.voxel_size).contiguous()], axis=-1).cuda()
                        # in_field = ME.TensorField(
                        #     features=torch.ones_like(coords),
                        #     coordinates=coords,
                        #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                        #     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                        #     # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
                        #     # device=in_dict['feats'].device,
                        # )
                        
                        # for i in range(100):
                        #     down = in_field.sparse(8)
                        #     print(down)
                        #     down_coords_2 = down._C[:,1:].float()
                        #     down_coords_2 = self.sort_coords(down_coords_2)
                        #     print(down_coords[:30], down_coords_2[:30])
                        #     assert(torch.allclose(down_coords, down_coords_2.cpu()))
                        # assert(True == False)


    

    def setup(self, stage: Optional[str] = None):
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
            self.train_files = f.readlines()
            self.train_files = [file[:-5] for file in self.train_files]
        # print(self.train_files)
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
            self.val_files = f.readlines()
            self.val_files = [file[:-5] for file in self.val_files]
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_test.txt'), 'r') as f:
            self.test_files = f.readlines()
            self.test_files = [file[:-5] for file in self.test_files]

    def load_sample(self, idx):
        if self.use_orig_pcs:
            scene_data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'scans_processed_full_normals', idx + '.npy')))
            labels = scene_data[:, 9].long()
            labels[labels == 156] = -1
        else:
            scene_data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'preprocessing', 'scannet_scenes', idx + '.npy')))
            labels = scene_data[:, 10].long()

        out_dict = {'pts': scene_data[:, :3],  # include xyz by default
                    'labels': labels,
                    'scene_name': idx,
                    } 

        if self.use_colors:
            out_dict['colors'] = scene_data[:, 3:6]

        if self.use_normals:
            out_dict['normals'] = scene_data[:, 6:9]

        if self.structure_feats is not None:
            out_dict['structure_feats'] = torch.load(os.path.join(self.data_dir, self.structure_feats, idx + '_feats.pt'))
        else:
            out_dict['structure_feats'] = None

        if self.color_feats is not None:
            out_dict['color_feats'] = torch.load(os.path.join(self.data_dir, self.color_feats, idx + '_feats.pt'))
        else:
            out_dict['color_feats'] = None

        if self.load_graph:
            out_dict['adj'] = torch.load(os.path.join(self.data_dir, 'adjs', idx + '_adj.pt'))

        return out_dict


    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePrecomputed.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetPrecomputed")
        return parent_parser
