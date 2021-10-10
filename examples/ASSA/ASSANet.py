import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List

from . import pointnet2_utils
from .conv import build_grouper, build_conv, build_activation_layer

class ASSANetSeg(LightningModule):
    def __init__(self, cfg):
        """ASSA-Net implementation for paper:
        Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning
        Args:
            cfg (dict): configuration
        """
        super().__init__()
        self.encoder = ASSANetEncoder(cfg)
        self.decoder = ASSANetDecoder(cfg)
        self.head = SceneSegHeadPointNet(cfg.data.num_classes, in_channles=cfg.model.fp_mlps[0][0])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz, features):
        l_xyz, l_features = self.encoder(xyz, features)
        return self.head(self.decoder(l_xyz, l_features))

    def convert_sync_batchnorm(self):
        return
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ASSANetSeg")
        return parent_parser

class SceneSegHeadPointNet(nn.Module):
    def __init__(self, num_classes, in_channles):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super(SceneSegHeadPointNet, self).__init__()
        self.head = nn.Sequential(nn.Conv1d(in_channles, 32, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(32, num_classes, kernel_size=1, bias=True))

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits

class ASSANetEncoder(nn.Module):
    def __init__(self, cfg):
        """ASSA-Net implementation for paper:
        Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning
        Args:
            cfg (dict): configuration
        """
        super().__init__()
        self.model_cfg = cfg.model
        self.local_aggregation_cfg = cfg.model.sa_config.local_aggregation
        self.SA_modules = nn.ModuleList()

        mlps = self.model_cfg.sa_config.get('mlps', None)
        if mlps is None:
            width = self.model_cfg.get('width', None)
            depth = self.model_cfg.get('depth', 2)
            layers = self.local_aggregation_cfg.get('layers', None)
            assert width is not None
            assert layers is not None
            mlps = [[[width] * layers]*depth,
                    [[width * depth] * layers]*depth,
                    [[width * depth ** 2] * layers]*depth,
                    [[width * depth ** 3] * layers]*depth]
            post_layers = self.local_aggregation_cfg.get('post_layers', 0)
            if post_layers == -1:
                self.local_aggregation_cfg.post_layers = self.local_aggregation_cfg.get('post_layers', layers//2)
            self.model_cfg.sa_config.mlps = mlps
            print(f'channels for the current model is modified to {self.model_cfg.sa_config.mlps}')

            # revise the radius, and nsample
            for i in range(len(self.model_cfg.sa_config.radius)):
                self.model_cfg.sa_config.radius[i] = self.model_cfg.sa_config.radius[i] + \
                                                     [self.model_cfg.sa_config.radius[i][-1]]*(depth-2)
            for i in range(len(self.model_cfg.sa_config.nsample)):
                self.model_cfg.sa_config.nsample[i] = self.model_cfg.sa_config.nsample[i] + [
                    self.model_cfg.sa_config.nsample[i][-1]] * (depth - 2)

        # build the first conv and local aggregations on the input points. (this is to be similar to close3d).
        width = self.model_cfg.sa_config.mlps[0][0][0]
        activation = self.model_cfg.get('activation', 'relu')
        self.conv1 = nn.Sequential(*[nn.Conv1d(self.model_cfg.in_channel, width, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(width),
                                     build_activation_layer(activation, inplace=True)])

        grouper_cfg = self.local_aggregation_cfg.get('grouper', edict())
        grouper_cfg.radius = self.model_cfg.sa_config.radius[0][0]
        grouper_cfg.nsample = self.model_cfg.sa_config.nsample[0][0]
        grouper_cfg.npoint = self.model_cfg.sa_config.npoints[0]

        conv_cfg = self.local_aggregation_cfg.get('conv', edict())
        conv_cfg.channels = [width] * 4 
        la1_cfg = edict(self.local_aggregation_cfg.copy())
        la1_cfg.post_layers = 1
        self.la1 = LocalAggregation(conv_cfg, grouper_cfg, la1_cfg)

        skip_channel_list = [width]
        for k in range(self.model_cfg.sa_config.npoints.__len__()):  # sample times
            # obtain the in_channels and output channels from the configuration
            channel_list = self.model_cfg.sa_config.mlps[k].copy()
            channel_out = 0
            for idx in range(channel_list.__len__()):
                channel_list[idx] = [width] + channel_list[idx]
                channel_out += channel_list[idx][-1]  # concatenate
                width = channel_list[idx][-1]
            width = channel_out

            # for each sample, may query points multiple times, the query radii and nsamples may be different
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=self.model_cfg.sa_config.npoints[k],
                    radii=self.model_cfg.sa_config.radius[k],
                    nsamples=self.model_cfg.sa_config.nsample[k],
                    channel_list=channel_list,
                    local_aggregation_cfg=self.local_aggregation_cfg,
                    sample_method=self.model_cfg.sa_config.get('sample_method', 'fps')
                )
            )
            skip_channel_list.append(channel_out)
        self.encoder_layers = len(self.SA_modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz, features):
        """
        Args:
        Returns:
        """
        l_xyz, l_features = [[] for _ in range(self.encoder_layers + 1)], [[] for _ in range(self.encoder_layers + 1)]

        # first SA layer for processing the whole xyz without subsampling
        l_xyz[0] = xyz
        l_features[0] = self.la1(xyz, xyz, self.conv1(features))

        # repeated SA encoder modules
        for i in range(self.encoder_layers):
            l_xyz[i + 1], l_features[i + 1] = self.SA_modules[i](l_xyz[i], l_features[i])

        return l_xyz, l_features

class ASSANetDecoder(nn.Module):
    def __init__(self, cfg):
        """ASSA-Net implementation for paper:
        Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning
        Args:
            cfg (dict): configuration
        """
        super().__init__()
        self.model_cfg = cfg.model
        self.local_aggregation_cfg = cfg.model.sa_config.local_aggregation
        self.SA_modules = nn.ModuleList()

        mlps = self.model_cfg.sa_config.get('mlps', None)
        if mlps is None:
            width = self.model_cfg.get('width', None)
            depth = self.model_cfg.get('depth', 2)
            layers = self.local_aggregation_cfg.get('layers', None)
            assert width is not None
            assert layers is not None
            mlps = [[[width] * layers]*depth,
                    [[width * depth] * layers]*depth,
                    [[width * depth ** 2] * layers]*depth,
                    [[width * depth ** 3] * layers]*depth]
            post_layers = self.local_aggregation_cfg.get('post_layers', 0)
            if post_layers == -1:
                self.local_aggregation_cfg.post_layers = self.local_aggregation_cfg.get('post_layers', layers//2)
            self.model_cfg.sa_config.mlps = mlps
            print(f'channels for the current model is modified to {self.model_cfg.sa_config.mlps}')

            # revise the radius, and nsample
            for i in range(len(self.model_cfg.sa_config.radius)):
                self.model_cfg.sa_config.radius[i] = self.model_cfg.sa_config.radius[i] + \
                                                     [self.model_cfg.sa_config.radius[i][-1]]*(depth-2)
            for i in range(len(self.model_cfg.sa_config.nsample)):
                self.model_cfg.sa_config.nsample[i] = self.model_cfg.sa_config.nsample[i] + [
                    self.model_cfg.sa_config.nsample[i][-1]] * (depth - 2)

        # build the first conv and local aggregations on the input points. (this is to be similar to close3d).
        width = self.model_cfg.sa_config.mlps[0][0][0]
        skip_channel_list = [width]
        for k in range(self.model_cfg.sa_config.npoints.__len__()):  # sample times

            # obtain the in_channels and output channels from the configuration
            channel_list = self.model_cfg.sa_config.mlps[k].copy()
            channel_out = 0
            for idx in range(channel_list.__len__()):
                channel_list[idx] = [width] + channel_list[idx]
                channel_out += channel_list[idx][-1]  # concatenate
                width = channel_list[idx][-1]
            width = channel_out
            skip_channel_list.append(channel_out)

        self.decoders = nn.ModuleList()
        for k in range(self.model_cfg.fp_mlps.__len__()):
            pre_channel = self.model_cfg.fp_mlps[k + 1][-1] if k + 1 < len(self.model_cfg.fp_mlps) else channel_out
            self.decoders.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.fp_mlps[k],
                    local_aggregation_cfg=self.local_aggregation_cfg
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, l_xyz, l_features):
        """
        Args:
        Returns:
        """
        # repeated decoder modules
        for i in range(-1, -(len(self.decoders) + 1), -1):
            l_features[i - 1] = self.decoders[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        return l_features[0]

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.depth = 1
        self.sample_method = 'fps'
        self.sampler = None
        self.local_aggregations = None

    def forward(self, support_xyz: torch.Tensor,
                support_features: torch.Tensor = None, query_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param support_xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param support_features: (B, N, C) tensor of the features
        :param query_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        if query_xyz is None:
            if self.npoint is not None:
                if self.sample_method.lower() == 'fps':
                    xyz_flipped = support_xyz.transpose(1, 2).contiguous()
                    idx = pointnet2_utils.furthest_point_sample(support_xyz, self.npoint)
                    query_xyz = pointnet2_utils.gather_operation(
                        xyz_flipped,
                        idx).transpose(1, 2).contiguous()
                elif self.sample_method.lower() == 'random':
                    query_xyz, idx = self.sampler(support_xyz, support_features)
                    idx = idx.to(torch.int32)
            else:
                query_xyz = support_xyz
                idx = None
                
        for i in range(self.depth):
            new_features = self.local_aggregations[i](query_xyz, support_xyz, support_features,
                                                      query_idx=idx)
            support_xyz = query_xyz
            support_features = new_features
            idx = None
            new_features_list.append(new_features)

        return query_xyz, torch.cat(new_features_list, dim=1)  # concatenate

class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping
        PointNet++ Set Abstraction Module:
        1. For each module, downsample the point cloud once
        2. For each downsampled point cloud, query neighbors from the HR point cloud multiple times
        3. In each neighbor querying, build the aggregation_features, perform local aggregations
    """

    def __init__(self,
                 npoint: int,
                 radii: List[float],
                 nsamples: List[int],
                 channel_list: List[List[int]],
                 local_aggregation_cfg: dict,
                 sample_method='fps'
                 ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(channel_list)  # time for querying and performing local aggregations

        self.npoint = npoint  # the number of sampled points
        self.depth = len(radii)
        self.sample_method = sample_method

        if self.sample_method.lower() == 'random':
            self.sampler = pointnet2_utils.DenseRandomSampler(num_to_sample=self.npoint)

        self.local_aggregation_cfg = local_aggregation_cfg

        # holder for the grouper and convs (MLPs, \etc)
        self.local_aggregations = nn.ModuleList()

        grouper_cfg = local_aggregation_cfg.get('grouper', edict())
        conv_cfg = local_aggregation_cfg.get('conv', edict())

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            channels = channel_list[i]

            grouper_cfg.radius = radius
            grouper_cfg.nsample = nsample
            grouper_cfg.npoint = npoint

            # build the convs
            conv_cfg.channels = channels
            self.local_aggregations.append(LocalAggregation(conv_cfg,
                                                            grouper_cfg,
                                                            self.local_aggregation_cfg))


class PointnetFPModule(nn.Module):
    r"""Propagates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True, local_aggregation_cfg):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        conv_cfg = local_aggregation_cfg.get('conv', edict())
        if conv_cfg['method'] == 'conv2d':  # use 1d
            conv_cfg['method'] = 'conv1d'
        conv_cfg.channels = mlp
        self.convs = build_conv(conv_cfg, last_act=True)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features. To upsample!!!
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            # can we use other poitshuffle for upsampling?
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = self.convs(new_features)

        return new_features


class LocalAggregation(nn.Module):
    def __init__(self,
                 conv_cfg,
                 grouper_cfg,
                 config):
        """LocalAggregation operators
        Args:
            config: config file
        """
        super(LocalAggregation, self).__init__()
        self.conv_cfg = conv_cfg
        self.grouper_cfg = grouper_cfg
        if config.type.lower() == 'preconv':
            self.SA_CONFIG_operator = PreConv(conv_cfg, grouper_cfg, config)
        elif config.type.lower() == 'convpool':
            self.SA_CONFIG_operator = ConvPool(conv_cfg, grouper_cfg, config)
        else:
            raise NotImplementedError(f'LocalAggregation {config.type.lower()} not implemented')

    def forward(self, query_xyz, support_xyz, support_features,
                query_mask=None, support_mask=None, query_idx=None):
        """
        Args:
        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.SA_CONFIG_operator(query_xyz, support_xyz, support_features, query_mask, support_mask, query_idx)


CHANNEL_MAP = {
    'fj': lambda x: x,
    'assa': lambda x: x,
    'dpf*fj': lambda x: x,
    'dp*dp_fj': lambda x: x + 3,
    'absdp*fj': lambda x: x,
    'dr*fj': lambda x: x,
    'aw*fj': lambda x: x,
    'sumdp*fj': lambda x: x,
    'xyz': lambda x: x,
    'sin_cos': lambda x: x,
    'dp_fj': lambda x: 3 + x,
    'dp*df': lambda x: x,
    'df*fj': lambda x: x,
    'fi_df': lambda x: 2 * x,
    'dp_fi_df': lambda x: 3 + 2 * x,
    'transformer': lambda x: x
}


class PreConv(nn.Module):
    def __init__(self,
                 conv_cfg,
                 grouper_cfg,
                 config):
        """A PreConv operator for local aggregation
        Args:
            config: config file
        """
        super(PreConv, self).__init__()
        self.feature_type = config.feature_type
        self.reduction = config.reduction
        self.post_res = config.get('post_res', False)
        self.pre_res = config.get('pre_res', False)  # use residual convs/mlps in preconv and postconv.

        post_layers = config.get('post_layers', 0)
        if post_layers == -1:
            post_layers = config.layers//2
        # build grouper
        group_method = grouper_cfg.get('method', 'ball_query').lower()
        self.use_mask = 'mask' in group_method
        self.grouper = build_grouper(grouper_cfg)
        self.nsample = grouper_cfg.get('nsample', 20)
        self.radius = grouper_cfg.get('radius', None)
        assert self.radius is not None

        # only supports conv1d in PreConv Module. 
        assert '1d' in conv_cfg['method']

        # Build PreConvs before Pooling
        channels = conv_cfg.channels
        pre_conv_cfg = conv_cfg.copy()
        if post_layers != 0:
            pre_conv_cfg['channels'] = pre_conv_cfg['channels'][:-(post_layers)]
        if self.feature_type == 'assa':
            # TODO: aviod information leak. 
            pre_conv_cfg['channels'][-1] = int(np.ceil(pre_conv_cfg['channels'][-1]/3.0))
        self.pre_conv = build_conv(pre_conv_cfg, last_act=not self.pre_res)

        # Build skip connection layer for the pre convs
        if self.pre_res:
            if pre_conv_cfg['channels'][-1] != pre_conv_cfg['channels'][0]:
                short_cut_cfg = conv_cfg.copy()
                short_cut_cfg['channels'] = [pre_conv_cfg['channels'][0], pre_conv_cfg['channels'][-1]]
                self.pre_short_cut = build_conv(short_cut_cfg, last_act=not self.pre_res)
            else:
                self.pre_short_cut = nn.Sequential()
        act_name = conv_cfg.get('activation', 'relu')
        self.pre_activation = build_activation_layer(act_name) if self.pre_res else nn.Sequential()

        # Build PostConvs 
        if post_layers != 0:
            post_conv_cfg = conv_cfg.copy()
            post_channel_in = CHANNEL_MAP[config.feature_type](pre_conv_cfg['channels'][-1])
            if self.feature_type == 'assa':
                post_channel_in *= 3
            post_conv_cfg['channels'] = [post_channel_in] + [channels[-1]] * post_layers
            self.post_conv = build_conv(post_conv_cfg, last_act=not self.post_res)
        else:
            self.post_conv = nn.Sequential()
        self.post_activation = build_activation_layer(act_name) if self.post_res else nn.Sequential()

        # Build skip connection layer for the post convs
        if self.post_res:
            if pre_conv_cfg['channels'][-1] != post_conv_cfg['channels'][-1]:
                short_cut_cfg = conv_cfg.copy()
                short_cut_cfg['channels'] = [pre_conv_cfg['channels'][-1], channels[-1]]
                self.post_short_cut = build_conv(short_cut_cfg, last_act=False)
            else:
                self.post_short_cut = nn.Sequential()

        # reduction layer
        if self.reduction == 'max':
            self.reduction_layer = lambda x: F.max_pool2d(
                x, kernel_size=[1, self.nsample]
            ).squeeze(-1)

        elif self.reduction == 'avg' or self.reduction == 'mean':
            self.reduction_layer = lambda x: torch.mean(x, dim=-1, keepdim=False)

        elif self.reduction == 'sum':
            self.reduction_layer = lambda x: torch.sum(x, dim=-1, keepdim=False)
        else:
            raise NotImplementedError(f'reduction {self.reduction} not implemented')

    def forward(self, query_xyz, support_xyz, features, query_mask, support_mask, query_idx=None):
        """
        Args:
        Returns:
           output features of query points: [B, C_out, 3]
        """
        # PreConv Layer with possible residual connection
        # print(f"input featurs, {features.shape}") 
        if self.pre_res:
            features = self.pre_conv(features) + self.pre_short_cut(features)
        else:
            features = self.pre_conv(features)
        features = self.pre_activation(features)
        # print(f"after PreConv, {features.shape}")

        # subsampling + grouping layer.
        # subsampling has already been executed outside this module.
        # here, we direcly use the precomputed the query_xyz and the idx
        neighborhood_features, relative_position = self.grouper(query_xyz, support_xyz, features)
        B, C, npoint, nsample = neighborhood_features.shape
        # print(f"after grouping, {neighborhood_features.shape}")

        # subsample layer if query_idx is not None.
        if query_idx is not None:
            # torch gather is slower than this c++ implementation
            # center_feature = torch.gather(features, 2, query_idx.unsqueeze(1).repeat(1, features.shape[1], 1))
            features = pointnet2_utils.gather_operation(features, query_idx)
        # print(f"after subsampling, {features.shape}")

        # Anisotropic Reduction layer
        neighborhood_features = neighborhood_features.unsqueeze(1).expand(-1, 3, -1, -1, -1) \
                                * relative_position.unsqueeze(2)
        neighborhood_features = neighborhood_features.view(B, -1, npoint, nsample)
        neighborhood_features = self.reduction_layer(neighborhood_features)
        # print(f"after reduction, {neighborhood_features.shape}")

        # Post Conv layer with possible residual connection
        if self.post_res:
            features = self.post_conv(neighborhood_features) + self.post_short_cut(features)
        else:
            features = self.post_conv(neighborhood_features)
        features = self.post_activation(features)
        # print(f"after post Convs, {features.shape}")
        return features


class ConvPool(nn.Module):
    def __init__(self,
                 conv_cfg,
                 grouper_cfg,
                 config):
        """A PosPool operator for local aggregation
        Args:
            out_channels: input channels.
            nsample: neighborhood limit.
            config: config file
        """
        super(ConvPool, self).__init__()
        self.feature_type = config.feature_type
        self.reduction = config.reduction

        # use conv2d is wrongly used.
        channel_in = CHANNEL_MAP[config.feature_type](conv_cfg['channels'][0])
        conv_cfg.channels[0] = channel_in
        if conv_cfg['method'] == 'conv1d':
            conv_cfg['method'] = 'conv2d'

        self.convs = build_conv(conv_cfg)

        # build grouper
        self.grouper = build_grouper(grouper_cfg)

    def forward(self, query_xyz, support_xyz, support_features, query_mask=None, support_mask=None, query_idx=None):
        """
        Args:
        Returns:
           output features of query points: [B, C_out, 3]
        """
        neighborhood_features, relative_position = self.grouper(query_xyz, support_xyz, support_features)

        B, C, npoint, nsample = neighborhood_features.shape

        if 'df' not in self.feature_type:
            if self.feature_type == 'assa':
                if C >= 3:
                    repeats = C // 3
                    repeat_tensor = torch.tensor([repeats, repeats, C - repeats * 2], dtype=torch.long,
                                                 device=relative_position.device, requires_grad=False)
                    position_embedding = torch.repeat_interleave(relative_position, repeat_tensor, dim=1)
                    aggregation_features = position_embedding * neighborhood_features  # (B, C//3, 3, npoint, nsample)
                else:
                    attn = torch.sum(relative_position, dim=1, keepdims=True)
                    aggregation_features = neighborhood_features * attn

            elif self.feature_type == 'sumdp*fj':
                attn = torch.sum(relative_position, dim=1, keepdims=True)
                aggregation_features = neighborhood_features * attn
            elif self.feature_type == 'sincos*fj':
                feat_dim = C // 6
                wave_length = 1000
                alpha = 100
                feat_range = torch.arange(feat_dim, dtype=torch.float32).to(neighborhood_features.device)
                dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
                position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
                div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
                sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
                cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
                position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
                position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
                position_embedding = position_embedding.view(B, C, npoint, nsample)  # (B, C, npoint, nsample)
                aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)

            elif self.feature_type == 'fj':
                aggregation_features = neighborhood_features

            elif self.feature_type == 'dp_fj':
                aggregation_features = torch.cat([relative_position, neighborhood_features], 1)

        else:
            center_feature = neighborhood_features[..., 0]
            center_features = torch.unsqueeze(center_feature, -1).repeat([1, 1, 1, nsample])
            relative_features = neighborhood_features - center_features

            if self.feature_type == 'df':
                aggregation_features = relative_features

            elif self.feature_type == 'fi_df':
                aggregation_features = torch.cat([center_features, relative_features], 1)

            elif self.feature_type == 'dp_fi_df':
                aggregation_features = torch.cat([relative_position, center_features, relative_features], 1)

        aggregation_features = self.convs(aggregation_features)

        if len(aggregation_features.size()) == 4:
            if self.reduction == 'max':
                out_features = F.max_pool2d(
                    aggregation_features, kernel_size=[1, nsample]
                ).squeeze(-1)

            elif self.reduction == 'avg' or self.reduction == 'mean':
                out_features = torch.mean(aggregation_features, dim=-1, keepdim=False)

            elif self.reduction == 'sum':
                out_features = torch.sum(aggregation_features, dim=-1, keepdim=False)
            else:
                raise NotImplementedError(f'reduction {self.reduction} not implemented')
        else:
            out_features = aggregation_features

        return out_features

def build_grouper(grouper_cfg):
    radius = grouper_cfg.get('radius', 0.1)
    nsample = grouper_cfg.get('nsample', 20)
    normalize_xyz = grouper_cfg.get('normalize_xyz', False)

    grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=False,
                                            ret_grouped_xyz=True, normalize_xyz=normalize_xyz)

    return grouper

def build_conv(conv_cfg, last_act=True):
    channels = conv_cfg.get('channels', None)
    assert channels is not None
    method = conv_cfg.get('method', 'mlp').lower()
    use_bn = conv_cfg.get('use_bn', True)
    act_name = conv_cfg.get('activation', 'relu')
    activation = build_activation_layer(act_name)
    groups = conv_cfg.get('groups', 1)
    shared_mlps = []
    if method == 'conv2d':
        for k in range(len(channels) - 1):
            shared_mlps.append(nn.Conv2d(channels[k], channels[k + 1], kernel_size=1, groups=groups, bias=False))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(channels[k + 1]))
            if k != len(channels) - 2 or last_act:
                shared_mlps.append(activation)

    elif method == 'conv1d':
        for k in range(len(channels) - 1):
            shared_mlps.append(nn.Conv1d(channels[k], channels[k + 1], kernel_size=1, groups=groups, bias=False))
            if use_bn:
                shared_mlps.append(nn.BatchNorm1d(channels[k + 1]))
            if k != len(channels) - 2 or last_act:
                shared_mlps.append(activation)
    else:
        raise NotImplementedError(f'{method} in local aggregation transform is not supported currently')

    return nn.Sequential(*shared_mlps)
