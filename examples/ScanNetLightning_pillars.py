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
import pickle

# import MinkowskiEngine as ME

# from examples.voxelizer import SparseVoxelizer
# import examples.transforms as t
from examples.str2bool import str2bool
from examples.ScanNetLightning import ScanNet
import torch_geometric
from examples.utils import sparse_collate, save_pc
# from examples.utils import get_embedder
from sklearn.neighbors import KDTree
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops'))
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


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

class ScanNetLIG_pillars(ScanNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_scans = glob.glob(os.path.join(self.scans_dir, '*'))
        self.scene_names = [scan.split('/')[-1] for scan in all_scans]
        self.scene_name_idx = dict(zip(self.scene_names, range(len(self.scene_names))))

    def setup(self, stage: Optional[str] = None):
        # super().setup(stage)
        self.train_files = np.arange(self.num_steps).tolist()
        # self.val_files = (np.arange(100) + self.num_steps - 100).tolist()
        filename = os.path.join(self.data_dir, 's3dis_preprocessing', f'{self.subsampling_parameter:.3f}_data.pkl')
        with open(filename, 'rb') as f:
            (self.clouds_points, self.clouds_points_feats, self.clouds_points_labels,
             self.sub_clouds_points, self.sub_clouds_points_feats, self.sub_clouds_points_labels,
             self.sub_cloud_trees) = pickle.load(f)
            print(f"{filename} loaded successfully")
        filename = os.path.join(self.data_dir, 's3dis_preprocessing',
                                f'{self.subsampling_parameter:.3f}_{self.num_epochs}_{self.num_steps}_iterinds.pkl')
        with open(filename, 'rb') as f:
            self.cloud_inds, self.point_inds, self.noise = pickle.load(f)
            print(f"{filename} loaded successfully")
        filename = os.path.join(self.data_dir, 's3dis_preprocessing', f'{self.subsampling_parameter:.3f}_proj.pkl')
        with open(filename, 'rb') as f:
            self.projections = pickle.load(f)
            print(f"{filename} loaded successfully")


    def s3dis_preprocessing(self, all_scans, split):
        filename = os.path.join(self.data_dir, 's3dis_preprocessing', f'{split}_{self.subsampling_parameter:.3f}_data.pkl')
        if not os.path.exists(filename):
            cloud_points_list, cloud_points_feat_list, cloud_points_label_list = [], [], []
            sub_cloud_points_list, sub_cloud_points_label_list, sub_cloud_points_feat_list = [], [], []
            sub_cloud_tree_list = []

            for cloud_idx, scene_name in enumerate(all_scans):
                # scene_name = cloud_name.split('/')[-1]
                # Pass if the cloud has already been computed
                cloud_file = os.path.join(self.data_dir, 's3dis_preprocessing', scene_name + '.pkl')
                if os.path.exists(cloud_file):
                    with open(cloud_file, 'rb') as f:
                        cloud_points, cloud_feats, cloud_classes = pickle.load(f)
                else:
                    # Get rooms of the current cloud
                    file_name = os.path.join(self.data_dir, 'scans', scene_name, scene_name + '_vh_clean_2.ply')
                    with open(file_name, 'rb') as f:
                        plydata = PlyData.read(f)
                    cloud_points = torch.from_numpy(np.stack((plydata['vertex']['x'],
                                           plydata['vertex']['y'],
                                           plydata['vertex']['z'])).T)

                    pts_min = cloud_points.min(axis=0)[0]
                    cloud_points -= pts_min
                    cloud_colors = np.stack((plydata['vertex']['red'],
                                       plydata['vertex']['green'],
                                       plydata['vertex']['blue'])).T
                    label_file = file_name[:-4] + '.labels.ply'
                    with open(label_file, 'rb') as f:
                        plydata = PlyData.read(f)
                    cloud_classes = np.array(plydata['vertex']['label'], dtype=np.uint8)
                    cloud_classes = torch.tensor([self.label_map[x] for x in cloud_classes], dtype=torch.long)
                    cloud_implicit_feats = self.load_implicit_feats(file_name, cloud_points)
                    cloud_points = cloud_points.numpy()
                    cloud_feats = np.concatenate([cloud_colors, cloud_implicit_feats], axis=1)
                    with open(cloud_file, 'wb') as f:
                        pickle.dump((cloud_points, cloud_feats, cloud_classes), f)
                cloud_points_list.append(cloud_points)
                cloud_points_feat_list.append(cloud_feats)
                cloud_points_label_list.append(cloud_classes)

                sub_cloud_file = os.path.join(self.data_dir, 's3dis_preprocessing', scene_name + f'_{self.subsampling_parameter:.3f}_sub.pkl')
                if os.path.exists(sub_cloud_file):
                    with open(sub_cloud_file, 'rb') as f:
                        sub_points, sub_feats, sub_labels, search_tree = pickle.load(f)
                else:
                    if self.subsampling_parameter > 0:
                        sub_points, sub_feats, sub_labels = grid_subsampling(cloud_points,
                                                                              features=cloud_feats,
                                                                              labels=cloud_classes,
                                                                              sampleDl=self.subsampling_parameter)
                        # sub_colors /= 255.0 
                        sub_labels = np.squeeze(sub_labels)
                    else:
                        sub_points = cloud_points
                        # sub_colors = cloud_colors / 255.0
                        sub_labels = cloud_classes

                    # Get chosen neighborhoods
                    search_tree = KDTree(sub_points, leaf_size=50)

                    with open(sub_cloud_file, 'wb') as f:
                        pickle.dump((sub_points, sub_feats, sub_labels, search_tree), f)

                sub_cloud_points_list.append(sub_points)
                sub_cloud_points_feat_list.append(sub_feats)
                sub_cloud_points_label_list.append(sub_labels)
                sub_cloud_tree_list.append(search_tree)

            with open(filename, 'wb') as f:
                pickle.dump((cloud_points_list, cloud_points_feat_list, cloud_points_label_list,
                             sub_cloud_points_list, sub_cloud_points_feat_list, sub_cloud_points_label_list,
                             sub_cloud_tree_list), f)
                print(f"{filename} saved successfully")
            

        # prepare iteration indices
        filename = os.path.join(self.data_dir, 's3dis_preprocessing',
                                f'{split}_{self.subsampling_parameter:.3f}_{self.num_epochs}_{self.num_steps}_iterinds.pkl')
        if not os.path.exists(filename):
            potentials = []
            min_potentials = []
            for cloud_i, tree in enumerate(self.sub_cloud_trees):
                print(f"{cloud_i} has {tree.data.shape[0]} points")
                cur_potential = np.random.rand(tree.data.shape[0]) * 1e-3
                potentials.append(cur_potential)
                min_potentials.append(float(np.min(cur_potential)))
            cloud_inds = []
            point_inds = []
            noises = []
            for ep in range(self.num_epochs):
                for st in tqdm(range(self.num_steps)):
                    cloud_ind = int(np.argmin(min_potentials))
                    point_ind = np.argmin(potentials[cloud_ind])
                    # print(f"[{ep}/{st}]: {cloud_ind}/{point_ind}")
                    cloud_inds.append(cloud_ind)
                    point_inds.append(point_ind)
                    points = np.array(self.sub_cloud_trees[cloud_ind].data, copy=False)
                    center_point = points[point_ind, :].reshape(1, -1)
                    noise = np.random.normal(scale=self.in_radius / 10, size=center_point.shape)
                    noises.append(noise)
                    pick_point = center_point + noise.astype(center_point.dtype)
                    # Indices of points in input region
                    query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                              r=self.in_radius,
                                                                              return_distance=True,
                                                                              sort_results=True)[0][0]
                    cur_num_points = query_inds.shape[0]
                    if self.num_points < cur_num_points:
                        query_inds = query_inds[:self.num_points]
                    # Update potentials (Tuckey weights)
                    dists = np.sum(np.square((points[query_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(self.in_radius))
                    tukeys[dists > np.square(self.in_radius)] = 0
                    potentials[cloud_ind][query_inds] += tukeys
                    min_potentials[cloud_ind] = float(np.min(potentials[cloud_ind]))
                    # print(f"====>potentials: {potentials}")
                    # print(f"====>min_potentials: {min_potentials}")
            with open(filename, 'wb') as f:
                pickle.dump((cloud_inds, point_inds, noises), f)
                print(f"{filename} saved successfully")
            

        # prepare validation projection inds
        filename = os.path.join(self.data_dir, 's3dis_preprocessing', f'{split}_{self.subsampling_parameter:.3f}_proj.pkl')
        if not os.path.exists(filename):
            proj_ind_list = []
            for points, search_tree in zip(self.clouds_points, self.sub_cloud_trees):
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_ind_list.append(proj_inds)
            # self.projections = proj_ind_list
            with open(filename, 'wb') as f:
                pickle.dump(proj_ind_list, f)
                print(f"{filename} saved successfully")

    def prepare_data(self):
        os.makedirs(os.path.join(self.data_dir, 's3dis_preprocessing'), exist_ok=True)
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_train.txt'), 'r') as f:
            self.train_files = f.readlines()
            self.train_files = [file[:-5] for file in self.train_files]
        self.s3dis_preprocessing(self.train_files, 'train')
        # print(self.train_files)
        with open(os.path.join(self.data_dir, 'splits', 'scannetv2_val.txt'), 'r') as f:
            self.val_files = f.readlines()
            self.val_files = [file[:-5] for file in self.val_files]
        self.s3dis_preprocessing(self.val_files, 'val')


    def load_ply(self, idx):
        # print(scene_name)
        # idx = self.scene_name_idx[scene_name]
        # print(idx)
        cloud_ind = self.cloud_inds[idx + self.trainer.current_epoch * self.num_steps]
        point_ind = self.point_inds[idx + self.trainer.current_epoch * self.num_steps]
        noise = self.noise[idx + self.trainer.current_epoch * self.num_steps]
        # cloud_ind = self.cloud_inds[idx]
        # point_ind = self.point_inds[idx]
        # noise = self.noise[idx]
        points = np.array(self.sub_cloud_trees[cloud_ind].data, copy=False)
        center_point = points[point_ind, :].reshape(1, -1)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Indices of points in input region
        query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                  r=self.in_radius,
                                                                  return_distance=True,
                                                                  sort_results=True)[0][0]
        # Number collected
        cur_num_points = query_inds.shape[0]
        if self.num_points < cur_num_points:
            # choice = np.random.choice(cur_num_points, self.num_points)
            # input_inds = query_inds[choice]
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            input_inds = query_inds[:self.num_points][shuffle_choice]
            mask = torch.ones(self.num_points).type(torch.bool)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            query_inds = query_inds[shuffle_choice]
            padding_choice = np.random.choice(cur_num_points, self.num_points - cur_num_points)
            input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            mask = torch.zeros(self.num_points).type(torch.bool)
            mask[:cur_num_points] = 1

        original_points = points[input_inds]
        current_points = (original_points - pick_point).astype(np.float32)
        current_points_height = original_points[:, 2:]
        current_points_height = torch.from_numpy(current_points_height).type(torch.float32)

        current_feats = self.sub_clouds_points_feats[cloud_ind][input_inds]
        # current_colors = (current_colors - self.color_mean) / self.color_std
        current_feats = torch.from_numpy(current_feats).type(torch.float32)

        # current_colors_drop = (torch.rand(1) > self.color_drop).type(torch.float32)
        # current_colors = (current_colors * current_colors_drop).type(torch.float32)
        current_points_labels = torch.from_numpy(self.sub_clouds_points_labels[cloud_ind][input_inds]).type(torch.int64)
        current_cloud_index = torch.from_numpy(np.array(cloud_ind)).type(torch.int64)

        current_points = current_points[mask]
        current_feats = current_feats[mask]
        current_points_labels = current_points_labels[mask]

        # save_pc(current_points, current_feats[:,:3], 'test_pc_s3dis.ply')
        # assert(True == False)

        out_dict = {'pts': current_points,
                    'colors': current_feats[:,:3],
                    'implicit_feats': current_feats[:,3:],
                    'labels': current_points_labels,
                    # 'scene_name': scene_name
                    }
        # output_list = [current_points, mask, current_feats,
        #                current_points_labels, current_cloud_index, input_inds]
        return out_dict

    def process_input(self, input_dict):
        input_dict['colors'] = (input_dict['colors'] / 255.) - 0.5
        input_dict['coords'] = input_dict['pts'] / self.voxel_size
        input_dict['feats'] = self.get_features(input_dict)
        input_dict['seg_feats'] = None
        input_dict['rand_shift'] = None
        return input_dict

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        coords_batch, feats_batch = sparse_collate(input_dict['coords'], input_dict['feats'],
                                                                          dtype=torch.float32)
        update_dict = {"coords": coords_batch,
                    "feats": feats_batch,
                    "idxs": idxs}
        input_dict.update(update_dict)
        return input_dict

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = ScanNet.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetLIG_pillars")
        parser.add_argument("--subsampling_parameter", type=float, default=0.04)
        parser.add_argument("--in_radius", type=float, default=2.0)
        parser.add_argument("--num_epochs", type=int, default=600)
        parser.add_argument("--num_steps", type=int, default=2000)
        parser.add_argument("--num_points", type=int, default=15000)
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