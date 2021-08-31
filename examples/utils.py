import torch

def interpolate_grid_feats(pts, grid, part_size=0.25):
    """Regular grid interpolator, returns inpterpolation coefficients.
    Args:
    pts: `[num_points, dim]` tensor, coordinates of points
    grid: `[b, *sizes, dim]` tensor of latents

    Returns:
    implicit feats: `[num_points, 2**dim * ( features + dim )]` tensor, neighbor
    latent codes and relative locations for each input point .
    """
    npts = pts.shape[0]
    dim = pts.shape[1]
    xmin = torch.min(pts, dim=0)[0] - (part_size * 1.1)
    half_part_size = part_size / 2
    # normalize coords for interpolation
    pts = (pts - xmin) / half_part_size

    # find neighbor indices
    ind0 = torch.floor(pts)  # `[num_points, dim]`
    ind1 = torch.ceil(pts)  # `[num_points, dim]`
    ind01 = torch.stack([ind0, ind1], dim=0)  # `[2, num_points, dim]`
    ind01 = torch.transpose(ind01, 1, 2)  # `[2, dim, num_points]`
    # print(grid.shape, ind1.max(dim=0)[0])

    # generate combinations for enumerating neighbors
    com_ = torch.stack(torch.meshgrid(*tuple(torch.tensor([[0,1]] * dim))), dim=-1)
    com_ = torch.reshape(com_, [-1, dim])  # `[2**dim, dim]`
    dim_ = torch.reshape(torch.arange(0,dim), [1, -1])
    dim_ = torch.tile(dim_, [2**dim, 1])  # `[2**dim, dim]`
    gather_ind = torch.stack([com_, dim_], dim=-1)  # `[2**dim, dim, 2]`
    ind_ = gather_nd(ind01, gather_ind)  # [2**dim, dim, num_pts]
    ind_n = torch.transpose(ind_, 0,2).transpose(1,2)  # neighbor indices `[num_pts, 2**dim, dim]`
    # if min_coord is not None:
    #     ind_n -= (min_coord-1)
    # print(grid.shape, ind1.max(dim=0)[0], ind1.min(dim=0)[0], min_coord)
    # print(grid.shape)
    grid = torch.nn.functional.pad(grid, (0,0,0,1,0,1,0,1), mode='constant', value=0)
    # print(grid.shape)
    lat = gather_nd(grid, ind_n) # `[num_points, 2**dim, in_features]`

    # weights of neighboring nodes
    xyz0 = ind0  # `[num_points, dim]`
    xyz1 = ind0 + 1  # `[num_points, dim]`
    xyz01 = torch.stack([xyz0, xyz1], dim=-1)  # [num_points, dim, 2]`
    xyz01 = torch.transpose(xyz01, 0,2)  # [2, dim, num_points]
    pos = gather_nd(xyz01, gather_ind)  # `[2**dim, dim, num_points]`
    pos = torch.transpose(pos, 0,2).transpose(1,2) # `[num_points, 2**dim, dim]`
    xloc = (torch.unsqueeze(pts, -2) - pos) # `[num_points, 2**dim, dim]`
    return lat, xloc

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

    # print(indices.shape, params.shape, indices.max(), indices.min())
    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    # print(indices.shape, params.shape, indices.max(), indices.min())
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