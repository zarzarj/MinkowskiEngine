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

from examples.ScanNetLightning import ScanNet

class ScanNetLIG(ScanNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = ScanNet.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ScanNetLIG")
        return parent_parser

    def process_input(self, input_dict):
        # print(input_dict)
        input_dict['coords'], input_dict['lats'] = input_dict['implicit_feats']
        if self.shift_coords and self.trainer.training:
            input_dict['rand_shift'] = (torch.rand(3) * 100).type_as(input_dict['coords'])
            input_dict['coords'] += input_dict['rand_shift']
        del input_dict['implicit_feats']
        return input_dict

    def load_implicit_feats(self, file_name, pts):
        scene_name = file_name.split('/')[-2]
        mask_file = os.path.join(self.data_dir, 'masks', scene_name+'-d1e-05-ps0.npy')
        lats_file = os.path.join(self.data_dir, 'lats', scene_name+'-d1e-05-ps0.npy')
        mask = torch.from_numpy(np.load(mask_file))
        lats = torch.from_numpy(np.load(lats_file))
        grid_range = [torch.arange(s) for s in mask.shape]
        grid =  torch.stack(torch.meshgrid(grid_range), dim=-1)
        coords = grid[mask]
        # sptensor = ME.SparseTensor(features=lats, coordinates=mask)
        return (coords, lats)

    def convert_batch(self, idxs):
        input_dict = self.load_scan_files(idxs)
        coords_batch, feats_batch = ME.utils.sparse_collate(input_dict['coords'],
                                                            input_dict['lats'],
                                                            dtype=torch.float32)
        out_dict = {"coords": coords_batch,
                    "feats": feats_batch,
                    "pts": input_dict['pts'],
                    "labels": input_dict['labels'],
                    }
        if self.shift_coords and self.trainer.training:
            out_dict["rand_shift"] = input_dict['rand_shift']
        return out_dict

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
    