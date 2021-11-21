import torch
import numpy as np
import copy
import collections.abc

def index_dict(in_dict, idx):
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = copy.deepcopy(v[idx])
        else:
            out_dict[k] = copy.deepcopy(v)
    return out_dict

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count



def coord_to_linear_idx(pts, dense_dim):
    linear_idx = pts[:,0] * dense_dim[1] * dense_dim[2] + \
                 pts[:,1] * dense_dim[2] + \
                 pts[:,2]
    return linear_idx.long()

def coord_mapping(tensor_coords):
    num_coords = tensor_coords.shape[0]
    dense_dim = tensor_coords.max(axis=0)[0]+2
    # print(dense_dim)
    tensor_coord_idx = -torch.ones(torch.prod(dense_dim), dtype=torch.long) 
    linear_idx = coord_to_linear_idx(tensor_coords, dense_dim)
    tensor_coord_idx[linear_idx] = torch.arange(linear_idx.shape[0], dtype=torch.long)
    return tensor_coord_idx, dense_dim

def sort_coords(coords):
    max_coords = coords.max(axis=0)[0]
    coords_hash = coords[:,0]
    for i in range(len(max_coords)-1):
        coords_hash = coords_hash * max_coords[i+1] + coords[:,i+1]
    sort_idx = coords_hash.sort()[1]
    coords = coords[sort_idx]
    return coords

def argsort_coords(coords):
    max_coords = coords.max(axis=0)[0]
    coords_hash = coords[:,0]
    for i in range(len(max_coords)-1):
        coords_hash = coords_hash * max_coords[i+1] + coords[:,i+1]
    sort_idx = coords_hash.sort()[1]
    return sort_idx

def features_at_coordinates(pts, coords, tensor_features):
    tensor_mapping, dense_dim = coord_mapping(coords)
    out_tensor = torch.zeros(pts.shape[0], tensor_features.shape[1]).type_as(tensor_features)
    pts_idx = tensor_mapping[coord_to_linear_idx(pts, dense_dim)]
    valid_pts = pts_idx != -1
    out_tensor[valid_pts] = tensor_features[pts_idx[valid_pts]]
    return out_tensor


def interpolate_sparsegrid_feats(pts, coords, feats, part_size=0.25, overlap_factor=2, xmin=None):
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
    # print(coords, pts, xmin)
    # if rand_shift is not None:
    #     pts += rand_shift

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
    # print(ind_n)
    xloc = torch.unsqueeze(pts, -2) - pos # `[num_points, 2**dim, dim]`
    if overlap_factor == 1:
        weights = torch.ones(xloc.shape[0], xloc.shape[1]).type_as(xloc)
    else:
        weights = torch.abs(torch.prod((overlap_factor - 1) - torch.abs(xloc), axis=-1))
    if overlap_factor > 1:
        weights /= (3 * (overlap_factor/2)**2 - overlap_factor)
    # print(lat.shape, xloc.shape, weights.shape)
    # print(xloc.min(), xloc.max())
    
    return lat, xloc, weights

def interpolate_grid_feats(pts, grid, part_size=0.25, overlap_factor=2, xmin=None, rand_shift=None):
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
    grid = torch.nn.functional.pad(grid, (0,0,0,1,0,1,0,1), mode='constant', value=0)
    # print(grid.shape)
    lat = gather_nd(grid, ind_n) # `[num_points, 2**dim, in_features]

    pos = ind_n.float() + 1 - overlap_factor/2.
    xloc = torch.unsqueeze(pts, -2) - pos # `[num_points, 2**dim, dim]`
    weights = torch.abs(torch.prod((overlap_factor - 1) - torch.abs(xloc), axis=-1))
    if overlap_factor > 1:
        weights /= (3 * (overlap_factor/2)**2 - overlap_factor)
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



def sparse_collate(coords, feats, labels=None, dtype=torch.int32, device=None):
    r"""Create input arguments for a sparse tensor `the documentation
    <https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html>`_.
    Convert a set of coordinates and features into the batch coordinates and
    batch features.
    Args:
        :attr:`coords` (set of `torch.Tensor` or `numpy.ndarray`): a set of coordinates.
        :attr:`feats` (set of `torch.Tensor` or `numpy.ndarray`): a set of features.
        :attr:`labels` (set of `torch.Tensor` or `numpy.ndarray`): a set of labels
        associated to the inputs.
    """
    use_label = False if labels is None else True
    feats_batch, labels_batch = [], []
    assert isinstance(
        coords, collections.abc.Sequence
    ), "The coordinates must be a sequence of arrays or tensors."
    assert isinstance(
        feats, collections.abc.Sequence
    ), "The features must be a sequence of arrays or tensors."
    D = np.unique(np.array([cs.shape[1] for cs in coords]))
    assert len(D) == 1, f"Dimension of the array mismatch. All dimensions: {D}"
    D = D[0]
    if device is None:
        if isinstance(coords, torch.Tensor):
            device = coords[0].device
        else:
            device = "cpu"
    assert dtype in [
        torch.int32,
        torch.float32,
    ], "Only torch.int32, torch.float32 supported for coordinates."

    if use_label:
        assert isinstance(
            labels, collections.abc.Sequence
        ), "The labels must be a sequence of arrays or tensors."

    N = np.array([len(cs) for cs in coords]).sum()
    Nf = np.array([len(fs) for fs in feats]).sum()
    assert N == Nf, f"Coordinate length {N} != Feature length {Nf}"

    batch_id = 0
    s = 0  # start index
    bcoords = torch.zeros((N, D + 1), dtype=dtype, device=device)  # uninitialized
    for coord, feat in zip(coords, feats):
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord)
        else:
            assert isinstance(
                coord, torch.Tensor
            ), "Coords must be of type numpy.ndarray or torch.Tensor"
        if dtype == torch.int32 and coord.dtype in [torch.float32, torch.float64]:
            coord = coord.floor()

        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        else:
            assert isinstance(
                feat, torch.Tensor
            ), "Features must be of type numpy.ndarray or torch.Tensor"

        # Labels
        if use_label:
            label = labels[batch_id]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            labels_batch.append(label)

        cn = coord.shape[0]
        # Batched coords
        bcoords[s : s + cn, 1:] = coord
        bcoords[s : s + cn, 0] = batch_id

        # Features
        feats_batch.append(feat)

        # Post processing steps
        batch_id += 1
        s += cn

    # Concatenate all lists
    feats_batch = torch.cat(feats_batch, 0)
    if use_label:
        if isinstance(labels_batch[0], torch.Tensor):
            labels_batch = torch.cat(labels_batch, 0)
        return bcoords, feats_batch, labels_batch
    else:
        return bcoords, feats_batch