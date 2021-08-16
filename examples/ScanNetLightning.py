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


import MinkowskiEngine as ME

from examples.voxelizer import SparseVoxelizer
import examples.transforms as t
from examples.str2bool import str2bool

class ScanNet(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
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
        }
        # self.cache = cache
        # self.cache_dict = defaultdict(dict)

        # Augmentation arguments
        # self.SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
        # self.ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
        # self.TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
        # self.ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
        # self.sparse_voxelizer = SparseVoxelizer(
        #     voxel_size=self.voxel_size,
        #     clip_bound=self.clip_bound,
        #     use_augmentation=self.augment_data,
        #     scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        #     rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        #     translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        #     rotation_axis=self.locfeat_idx,
        #     ignore_label=self.ignore_label)


        self.NUM_LABELS = 150  # Will be converted to 20 as defined in IGNORE_LABELS.
        self.IGNORE_LABELS = tuple(set(range(self.NUM_LABELS)) - set(self.valid_class_ids))
        # map labels not evaluated to ignore_label
        self.label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                self.label_map[l] = -100
            else:
                self.label_map[l] = n_used
                n_used += 1
        self.label_map[-100] = -100
        self.NUM_LABELS -= len(self.IGNORE_LABELS)

        # input_transforms = []
        # if self.augment_data:
        #     input_transforms += [
        #         t.RandomDropout(0.2),
        #         t.RandomHorizontalFlip(self.rotation_axis, False),
        #         t.ChromaticAutoContrast(),
        #         t.ChromaticTranslation(self.data_aug_color_trans_ratio),
        #         t.ChromaticJitter(self.data_aug_color_jitter_std),
        #     ]

        # if len(input_transforms) > 0:
        #     self.input_transforms = t.Compose(input_transforms)
        # else:
        #     self.input_transforms = None

        # if self.return_transformation:
        #     self.collate_fn = t.cflt_collate_fn_factory(self.limit_numpoints)
        # else:
        #     self.collate_fn = t.cfl_collate_fn_factory(self.limit_numpoints)


    # def _augment_elastic_distortion(self, pointcloud):
    #     if self.ELASTIC_DISTORT_PARAMS is not None:
    #       if random.random() < 0.95:
    #         for granularity, magnitude in self.ELASTIC_DISTORT_PARAMS:
    #           pointcloud = t.elastic_distortion(pointcloud, granularity, magnitude)
    #     return pointcloud

    def prepare_data(self):
        if self.save_preds:
            os.makedirs(os.path.join(self.data_dir, 'output'), exist_ok=True)
        train_filename = os.path.join(self.data_dir, 'train_idx.npy')
        val_filename = os.path.join(self.data_dir, 'val_idx.npy')
        if not os.path.exists(train_filename) or not os.path.exists(val_filename):
            scan_files = glob.glob(os.path.join(self.scans_dir, '*'))
            idxs = np.random.permutation(len(scan_files))
            num_training = math.ceil(idxs.shape[0] * self.train_percent)
            train_idx, val_idx = idxs[:num_training], idxs[num_training:]
            np.save(train_filename, train_idx)
            np.save(val_filename, val_idx)
        # scan_test_filenames = glob.glob(os.path.join(self.scans_test_dir, '*'))

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage =='validate':
            self.scan_files = glob.glob(os.path.join(self.scans_dir, '*', '*_vh_clean_2.ply'))
            self.train_idx = torch.from_numpy(np.load(os.path.join(self.data_dir, 'train_idx.npy')))
            self.val_idx = torch.from_numpy(np.load(os.path.join(self.data_dir, 'val_idx.npy')))
        else:
            self.scan_files = glob.glob(os.path.join(self.scans_test_dir, '*', '_vh_clean_2.ply'))
            self.test_idx = torch.from_numpy(np.arange(len(self.scan_files)))
        self.scan_files.sort()

        if self.preload and self.in_memory:
            t = time.perf_counter()
            print('Reading dataset...', end=' ', flush=True)
            # print(self.scan_files)
            input_dict = self.load_scan_files(np.arange(len(self.scan_files)))
            self.coords = input_dict['coords']
            self.colors = input_dict['colors']
            self.labels = input_dict['labels']
            if self.use_implicit_feats:
                self.implicit_feats = input_dict['implicit_feats']
            # print(self.coords)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        else:
            self.loaded = np.zeros(len(self.scan_files), dtype=np.bool)
            self.coords = [None]*len(self.scan_files)
            self.colors = [None]*len(self.scan_files)
            self.labels = [None]*len(self.scan_files)
            if self.use_implicit_feats:
                self.implicit_feats = [None]*len(self.scan_files)

        if self.use_coord_pos_encoding:
            self.embedder, _ = get_embedder(self.coord_pos_encoding_multires)
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_idx, collate_fn=self.convert_batch,
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_idx, collate_fn=self.convert_batch,
                          batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self):  # Test best validation model once again.
        return DataLoader(self.test_idx, collate_fn=self.convert_batch,
                          batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        feats = self.get_features(input_dict)
        coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(input_dict['coords'],
                                                                          feats, input_dict['labels'],
                                                                          dtype=torch.float32)
        # print(coords_batch.shape, feats_batch.shape, labels_batch.shape)
        # print(coords_batch, feats_batch, labels_batch)
        return {"coords": coords_batch,
                "feats": feats_batch,
                "labels": labels_batch,
                }

    def load_scan_files(self, idxs):
        coords = []
        colors = []
        labels = []
        if self.use_implicit_feats:
            implicit_feats = []
        for i in idxs:
            input_dict = self.load_ply(i)
            if self.use_implicit_feats:
                implicit_feats.append(input_dict['implicit_feats'])
            coords.append(input_dict['coords'])
            colors.append(input_dict['colors'])
            labels.append(input_dict['labels'])
        out_dict = {'coords': coords,
                    'colors': colors,
                    'labels': labels,
                    }
        if self.use_implicit_feats:
            out_dict['implicit_feats'] = implicit_feats
        return out_dict

    def load_ply_file(self, file_name):
        with open(file_name, 'rb') as f:
            plydata = PlyData.read(f)
        coords = np.stack((plydata['vertex']['x'],
                           plydata['vertex']['y'],
                           plydata['vertex']['z'])).T
        colors = np.stack((plydata['vertex']['red'],
                           plydata['vertex']['green'],
                           plydata['vertex']['blue'])).T
        coords /= self.voxel_size 
        colors = (colors / 255.) - 0.5
        return torch.from_numpy(coords), torch.from_numpy(colors)

    def load_ply_label_file(self, file_name):
        with open(file_name, 'rb') as f:
            plydata = PlyData.read(f)
        labels = np.array(plydata['vertex']['label'], dtype=np.uint8)
        labels = np.array([self.label_map[x] for x in labels], dtype=np.int)
        # print(labels)
        # print(labels.max(), labels.min())
        return torch.from_numpy(labels)

    def load_ply(self, idx):
        if self.in_memory and self.loaded[idx]:
            coords = self.coords[idx]
            colors = self.colors[idx]
            labels = self.labels[idx]
            if self.use_implicit_feats:
                implicit_feats = self.implicit_feats[idx]
        else:
            scan_file = self.scan_files[idx]
            coords, colors = self.load_ply_file(scan_file)
            # print(self.trainer.training, self.trainer.validating)
            if self.trainer.training or self.trainer.validating:
                label_file = scan_file[:-4] + '.labels.ply'
                labels = self.load_ply_label_file(label_file)
            else:
                labels = np.zeros(coords.shape[0])
            self.coords[idx] = coords
            self.colors[idx] = colors
            self.labels[idx] = labels
            if self.use_implicit_feats:
                implicit_feats = self.load_impicit_feats(scan_file, coords)
                self.implicit_feats[idx] = implicit_feats
            self.loaded[idx] = True
        out_dict = {'coords': coords,
                    'colors': colors,
                    'labels': labels,
                    }
        if self.use_implicit_feats:
            out_dict['implicit_feats'] = implicit_feats
        return out_dict

    def get_features(self, input_dict):
        feats = []
        if self.use_colors:
            feats.append(input_dict['colors'])
        if self.use_coords:
            feats.append(input_dict['coords'])
        if self.use_implicit_feats:
            feats.append(input_dict['implicit_feats'])
        if self.use_coord_pos_encoding:
            feats.append([self.embedder(coord) for coord in input_dict['coords']])
        out_feats = []
        for i in range(len(feats[0])):
            cur_all_feats = [feat[i] for feat in feats]
            # print(cur_all_feats, cur_a)
            out_feats.append(torch.cat(cur_all_feats, dim=-1))
        return out_feats


    def load_impicit_feats(self, file_name, pts):
        scene_name = file_name.split('/')[-2]
        implicit_feat_file = os.path.join(self.data_dir, 'implicit_feats', scene_name+'-d1e-05-ps0.pt')
        if not os.path.exists(implicit_feat_file):
            os.makedirs(os.path.join(self.data_dir, 'implicit_feats'), exist_ok=True)
            mask_file = os.path.join(self.data_dir, 'masks', scene_name+'-d1e-05-ps0.npy')
            lats_file = os.path.join(self.data_dir, 'lats', scene_name+'-d1e-05-ps0.npy')
            mask = np.load(mask_file)
            lats = np.load(lats_file)
            implicit_feats = get_implicit_feats(pts, lats, mask)
            torch.save(implicit_feats, implicit_feat_file)
        else:
            implicit_feats = torch.load(implicit_feat_file)
        return implicit_feats

    # def getitem(self, idx):
    #     if self.explicit_rotation > 1:
    #         rotation_space = np.linspace(-np.pi, np.pi, self.explicit_rotation + 1)
    #         rotation_angle = rotation_space[index % self.explicit_rotation]
    #         index //= self.explicit_rotation
    #     else:
    #         rotation_angle = None
    #     coords, feats, labels = self.load_ply(idx)
    #     if self.prevoxelize_voxel_size is not None:
    #         inds = ME.SparseVoxelize(coords[:, :3] / self.prevoxelize_voxel_size, return_index=True)
    #         coords = coords[inds]
    #         feats = feats[inds]
    #         labels = labels[inds]

    #     if self.elastic_distortion:
    #         pointcloud = self._augment_elastic_distortion(pointcloud)

    #     outs = self.sparse_voxelizer.voxelize(
    #         coords,
    #         feats,
    #         labels,
    #         center=np.array([0,0,0]),
    #         rotation_angle=rotation_angle,
    #         return_transformation=self.return_transformation)

    #     if self.return_transformation:
    #         coords, feats, labels, transformation = outs
    #         transformation = np.expand_dims(transformation, 0)
    #     else:
    #         coords, feats, labels = outs

    #     # map labels not used for evaluation to ignore_label
    #     if self.input_transforms is not None:
    #         coords, feats, labels = self.input_transforms(coords, feats, labels)
    #     if self.IGNORE_LABELS is not None:
    #         labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

    #     return_args = [coords, feats, labels]
    #     if self.return_transformation:
    #         return_args.extend([pointcloud.astype(np.float32), transformation.astype(np.float32)])
    #     return tuple(return_args)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ScanNet")
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=6)
        parser.add_argument("--test_batch_size", type=int, default=6)
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--in_memory", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--save_preds", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--preload", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--train_percent", type=float, default=0.8)
        # parser.add_argument("--augment_data", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("--return_transformation", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--voxel_size", type=float, default=0.02)
        # parser.add_argument("--rotation_axis", type=str, default='z')
        # parser.add_argument("--locfeat_idx", type=int, default=2)
        # parser.add_argument("--prevoxelize_voxel_size", type=float, default=None)
        # parser.add_argument("--clip_bound", type=float, default=None)
        # parser.add_argument("--data_aug_color_trans_ratio", type=float, default=0.10)
        # parser.add_argument("--data_aug_color_jitter_std", type=float, default=0.05)
        # parser.add_argument("--limit_numpoints", type=int, default=0)
        # parser.add_argument("--explicit_rotation", type=int, default=-1)
        # parser.add_argument("--elastic_distortion", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_implicit_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_coords", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_colors", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_coord_pos_encoding", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--coord_pos_encoding_multires", type=int, default=10)
        return parent_parser

    def cleanup(self):
        self.sparse_voxelizer.cleanup()



def get_implicit_feats(pts, lats, mask, part_size=0.25):
    """Regular grid interpolator, returns inpterpolation coefficients.
    Args:
    pts: `[num_points, dim]` tensor, coordinates of points
    lats: `[num_nonempty_lats, dim]` sparse tensor of latents
    mask: `[*size]` mask for nonempty latents

    Returns:
    implicit feats: `[num_points, 2**dim * ( features + dim )]` tensor, neighbor
    latent codes and relative locations for each input point .
    """
    # get dimensions
    size = torch.from_numpy(np.array(mask.shape))
    xmin = torch.min(pts[:, :3], 0)[0]
    xmin -= part_size
    true_shape = (size - 1) / 2.0
    xmax = xmin + true_shape * part_size

    grid = torch.zeros(mask.shape + (lats.shape[-1],), dtype=torch.float32)
    # print(grid.shape)
    mask = torch.from_numpy(mask).bool()
    grid[mask] = torch.from_numpy(lats)

    npts = pts.shape[0]
    cubesize = 1.0/(size-1.0)
    dim = len(size)

    # normalize coords for interpolation
    bbox = xmax - xmin
    pts = (pts - xmin) / bbox
    pts = torch.clip(pts, 1e-6, 1-1e-6)  # clip to boundary of the bbox

    # find neighbor indices
    ind0 = torch.floor(pts / cubesize)  # `[num_points, dim]`
    ind1 = torch.ceil(pts / cubesize)  # `[num_points, dim]`
    ind01 = torch.stack([ind0, ind1], dim=0)  # `[2, num_points, dim]`
    ind01 = torch.transpose(ind01, 1, 2)  # `[2, dim, num_points]`

    # generate combinations for enumerating neighbors
    com_ = torch.stack(torch.meshgrid(*tuple(torch.tensor([[0,1]] * dim))), dim=-1)
    com_ = torch.reshape(com_, [-1, dim])  # `[2**dim, dim]`
    # print(com_, com_.shape)
    dim_ = torch.reshape(torch.arange(0,dim), [1, -1])
    dim_ = torch.tile(dim_, [2**dim, 1])  # `[2**dim, dim]`
    # print(dim_, dim_.shape)
    gather_ind = torch.stack([com_, dim_], dim=-1)  # `[2**dim, dim, 2]`
    # print(gather_ind, gather_ind.shape)
    ind_ = gather_nd(ind01, gather_ind)  # [2**dim, dim, num_pts]
    # print(ind_, ind_.shape)
    ind_n = torch.transpose(ind_, 0,2).transpose(1,2)  # neighbor indices `[num_pts, 2**dim, dim]`
    # print(ind_n, ind_n.shape)
    lat = gather_nd(grid, ind_n) # `[num_points, 2**dim, in_features]`
    # print(lat, lat.shape)

    # weights of neighboring nodes
    xyz0 = ind0 * cubesize  # `[num_points, dim]`
    xyz1 = (ind0 + 1) * cubesize  # `[num_points, dim]`
    xyz01 = torch.stack([xyz0, xyz1], dim=-1)  # [num_points, dim, 2]`
    xyz01 = torch.transpose(xyz01, 0,2)  # [2, dim, num_points]
    pos = gather_nd(xyz01, gather_ind)  # `[2**dim, dim, num_points]`
    pos = torch.transpose(pos, 0,2).transpose(1,2) # `[num_points, 2**dim, dim]`
    xloc = (torch.unsqueeze(pts, -2) - pos) / cubesize # `[num_points, 2**dim, dim]`

    xloc = xloc.contiguous().reshape(npts, -1)
    lat = lat.contiguous().reshape(npts, -1)
    implicit_feats = torch.cat([lat, xloc], dim=-1) # `[num_points, 2**dim * (in_features + dim)]`
    # print(implicit_feats, implicit_feats.shape)
    return implicit_feats


def gather_nd(params, indices):
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)
    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices] 
    return output.reshape(out_shape).contiguous()


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=10):
    if multires == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim