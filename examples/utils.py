import torch
import numpy as np
import copy

def coord_to_linear_idx(pts, dense_dim):
    linear_idx = pts[:,0] * dense_dim[1] * dense_dim[2] + \
                 pts[:,1] * dense_dim[2] + \
                 pts[:,2]
    return linear_idx.long()

def coord_mapping(tensor_coords):
    num_coords = tensor_coords.shape[0]
    dense_dim = tensor_coords.max(axis=0)[0]+2
    tensor_coord_idx = -torch.ones(torch.prod(dense_dim), dtype=torch.long) 
    linear_idx = coord_to_linear_idx(tensor_coords, dense_dim)
    tensor_coord_idx[linear_idx] = torch.arange(linear_idx.shape[0], dtype=torch.long)
    return tensor_coord_idx, dense_dim

def features_at_coordinates(pts, coords, tensor_features):
    tensor_mapping, dense_dim = coord_mapping(coords)
    out_tensor = torch.zeros(pts.shape[0], tensor_features.shape[1]).type_as(tensor_features)
    pts_idx = tensor_mapping[coord_to_linear_idx(pts, dense_dim)]
    valid_pts = pts_idx != -1
    out_tensor[valid_pts] = tensor_features[pts_idx[valid_pts]]
    return out_tensor


def interpolate_sparsegrid_feats(pts, coords, feats, part_size=0.25, overlap_factor=2, xmin=None, rand_shift=None):
    """Regular grid interpolator, returns inpterpolation coefficients.
    Args:
    pts: `[num_points, dim]` tensor, coordinates of points
    grid: `[b, *sizes, dim]` tensor of latents

    Returns:
    implicit feats: `[num_points, 2**dim * ( features + dim )]` tensor, neighbor
    latent codes and relative locations for each input point .
    """
    # np.save('test_pts.npy', pts.cpu().numpy())
    pts = copy.deepcopy(pts)
    npts = pts.shape[0]
    dim = pts.shape[1]
    if xmin is not None:
        xmin -= (part_size * 1.1)
    else:
        xmin = -(part_size * 1.1)
    half_part_size = part_size / overlap_factor
    # normalize coords for interpolation
    pts = (pts - xmin) / half_part_size
    if rand_shift is not None:
        pts += rand_shift

    # find neighbor indices
    ind0 = torch.floor(pts)  # `[num_points, dim]`
    inds = [ind0 + i for i in range(overlap_factor)]
    ind01 = torch.stack(inds, dim=0)  # `[2, num_points, dim]`
    ind01 = torch.transpose(ind01, 1, 2)  # `[2, dim, num_points]`
    com_ = torch.stack(torch.meshgrid(*tuple(torch.tensor([np.arange(overlap_factor)] * dim))), dim=-1)
    com_ = torch.reshape(com_, [-1, dim])  # `[2**dim, dim]`
    dim_ = torch.reshape(torch.arange(0,dim), [1, -1])
    dim_ = torch.tile(dim_, [overlap_factor**dim, 1])  # `[2**dim, dim]`
    gather_ind = torch.stack([com_, dim_], dim=-1)  # `[2**dim, dim, 2]`
    ind_ = gather_nd(ind01, gather_ind)  # [2**dim, dim, num_pts]
    ind_n = torch.transpose(ind_, 0,2).transpose(1,2)  # neighbor indices `[num_pts, 2**dim, dim]`
    ind_m = ind_n.reshape(-1, dim)

    lat = features_at_coordinates(ind_m, coords, feats).reshape(ind_n.shape[0], ind_n.shape[1], -1)
    pos = ind_n.float() + 1 - overlap_factor/2.
    xloc = torch.unsqueeze(pts, -2) - pos # `[num_points, 2**dim, dim]`
    weights = torch.abs(torch.prod((overlap_factor - 1) - torch.abs(xloc), axis=-1))
    # print(xloc.min(), xloc.max())
    
    return lat, xloc, weights

def interpolate_grid_feats(pts, grid, part_size=0.25, overlap_factor=2):
    """Regular grid interpolator, returns inpterpolation coefficients.
    Args:
    pts: `[num_points, dim]` tensor, coordinates of points
    grid: `[b, *sizes, dim]` tensor of latents

    Returns:
    implicit feats: `[num_points, 2**dim * ( features + dim )]` tensor, neighbor
    latent codes and relative locations for each input point .
    """
    # np.save('test_pts.npy', pts.cpu().numpy())
    pts = copy.deepcopy(pts)
    npts = pts.shape[0]
    dim = pts.shape[1]
    xmin = torch.min(pts, dim=0)[0] - (part_size * 1.1)
    half_part_size = part_size / overlap_factor
    # normalize coords for interpolation
    pts = (pts - xmin) / half_part_size
    if overlap_factor != 2:
        pts-= overlap_factor - 1
    # find neighbor indices
    ind0 = torch.floor(pts)  # `[num_points, dim]`
    inds = [ind0 + i for i in range(overlap_factor)]
    ind01 = torch.stack(inds, dim=0)  # `[2, num_points, dim]`
    ind01 = torch.transpose(ind01, 1, 2)  # `[2, dim, num_points]`
    com_ = torch.stack(torch.meshgrid(*tuple(torch.tensor([np.arange(overlap_factor)] * dim))), dim=-1)
    com_ = torch.reshape(com_, [-1, dim])  # `[2**dim, dim]`
    dim_ = torch.reshape(torch.arange(0,dim), [1, -1])
    dim_ = torch.tile(dim_, [overlap_factor**dim, 1])  # `[2**dim, dim]`
    gather_ind = torch.stack([com_, dim_], dim=-1)  # `[2**dim, dim, 2]`
    ind_ = gather_nd(ind01, gather_ind)  # [2**dim, dim, num_pts]
    ind_n = torch.transpose(ind_, 0,2).transpose(1,2)  # neighbor indices `[num_pts, 2**dim, dim]`
    grid = torch.nn.functional.pad(grid, (0,0,0,1,0,1,0,1), mode='constant', value=0)
    # print(grid.shape)
    lat = gather_nd(grid, ind_n) # `[num_points, 2**dim, in_features]

    if overlap_factor == 2:
        pos = ind_n.float()
    else:
        pos = ind_n.float() + 1 - overlap_factor/2.
    xloc = torch.unsqueeze(pts, -2) - pos # `[num_points, 2**dim, dim]`
    weights = torch.abs(torch.prod((overlap_factor - 1) - torch.abs(xloc), axis=-1))
    # print(xloc.min(), xloc.max())
    return lat, xloc, weights

def gather_nd(params, indices):
    orig_shape = list(indices.shape)
    num_samples = torch.prod(torch.tensor(orig_shape[:-1]))
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


def save_pc(PC, PC_color, filename):
    from plyfile import PlyElement, PlyData
    PC = np.concatenate((PC, PC_color), axis=1)
    PC = [tuple(element) for element in PC]
    el = PlyElement.describe(np.array(PC, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([el]).write(filename)