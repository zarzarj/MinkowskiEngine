import warnings

import torch.nn as nn

from .activation import build_activation_layer, activation_str2dict_mapping
from .conv import build_conv_layer
from .norm import build_norm_layer, norm_str2dict_mapping
from .dropout import build_dropout_layer, drop_float2cfg_mapping
from .weight_init import kaiming_init, constant_init


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.
    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str):
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').

    Example:
        >>> # ---------------- Convolution
        >>> n_feats = 512
        >>> n_points = 128
        >>> n_batch = 4
        >>> device = torch.device('cuda')
        >>> feats = torch.rand((n_batch, n_feats, n_points, 1), dtype=torch.float).to(device) - 0.5
        >>> print(f"before operation, "
        >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")
        >>>
        >>> conv_cfg = {'type': "Conv2d"}
        >>> norm_cfg = {'type': 'BN2d', 'eps': 1e-5}
        >>> act_cfg = {'type': 'LeakyReLU'}
        >>> drop_cfg = {'type': 'Dropout2d'}
        >>>
        >>> conv = ConvModule(512, 32, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, drop_cfg=drop_cfg)
        >>> conv = conv.to(device)
        >>> # test convolution
        >>> feats = conv(feats)
        >>>
        >>> print(f"after operation, "
        >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 drop_cfg=None,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act', 'drop')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        assert drop_cfg is None or isinstance(drop_cfg, dict)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm

        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 4
        assert set(order) == set(['conv', 'norm', 'act', 'drop'])  # convolution -> normalization -> activation -> drop

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.with_dropout = drop_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        if conv_cfg['type'] == 'Linear':  # if linear layer, then only pass linear
            self.conv = build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                bias=bias
            )
            self.in_channels = self.conv.in_features
            self.out_channels = self.conv.out_features

        else:
            self.conv = build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode
            )
            # export the attributes of self.conv to a higher level for convenience
            self.in_channels = self.conv.in_channels
            self.out_channels = self.conv.out_channels
            self.kernel_size = self.conv.kernel_size
            self.stride = self.conv.stride
            self.padding = padding
            self.dilation = self.conv.dilation
            self.transposed = self.conv.transposed
            self.output_padding = self.conv.output_padding
            self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)  # Spectral normalization for GAN.

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = build_norm_layer(norm_cfg, norm_channels)

        # build activation layer
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)

        # build the dropout layer
        if self.with_dropout:
            self.dropout = build_dropout_layer(drop_cfg)
        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners, and we do not want ConvModule to
        #    overrides the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners, they will be initialized by this method with default
        #    `kaiming_init`.
        # 3. For PyTorch's conv layers, they will be initialized anyway by
        #    their own `reset_parameters` methods.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True, drop=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
            elif layer == 'drop' and drop and self.with_dropout:
                x = self.dropout(x)
        return x


class ConvModules(nn.Sequential):
    """a block consisted of sequential conv modules
    This block simplifies the usage of sequential convolution layers
    Args:
        channels_list (list of int): a list Number of channels for the input, (hidden layers), and output
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str):
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
        last_lin: Set the last layer as a pure linear layer without normalization, activation, dropout
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 channels_list,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 drop_cfg=None,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act', 'drop'),
                 last_lin=False,
                 ):

        m = []
        for i in range(1, len(channels_list)):
            if (i == len(channels_list) - 1) and last_lin:
                m.append(ConvModule(channels_list[i - 1], channels_list[i],
                                    kernel_size, stride, padding, dilation, groups, bias,
                                    conv_cfg, None, None, None,
                                    inplace, with_spectral_norm, padding_mode, order
                                    ))
            else:
                m.append(ConvModule(channels_list[i - 1], channels_list[i],
                                    kernel_size, stride, padding, dilation, groups, bias,
                                    conv_cfg, norm_cfg, act_cfg, drop_cfg,
                                    inplace, with_spectral_norm, padding_mode, order
                                    ))
        super(ConvModules, self).__init__(*m)


class Conv2d1x1(ConvModules):
    """Basic convolution modules for Dense input. [B, C, N, 1].
        Simply inherited from ConvModules.
    """

    def __init__(self,
                 channels_list,
                 act='relu',
                 norm='bn',
                 drop=0.,
                 last_lin=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='Conv2d'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act', 'drop'),
                 ):
        act_cfg = activation_str2dict_mapping(act)
        norm_cfg = norm_str2dict_mapping(norm)
        drop_cfg = drop_float2cfg_mapping(drop, 'Dropout2d')
        super(Conv2d1x1, self).__init__(channels_list, kernel_size, stride, padding, dilation, groups, bias,
                                        conv_cfg, norm_cfg, act_cfg, drop_cfg, inplace, with_spectral_norm,
                                        padding_mode, order, last_lin
                                        )


class LinearModules(ConvModules):
    """Basic convolution modules for Dense input. [B, C, N, 1].
        Simply inherited from ConvModules.
    """

    def __init__(self,
                 channels_list,
                 act='relu',
                 norm='bn1d',
                 drop=0.,
                 last_lin=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='Linear'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act', 'drop'),
                 ):
        act_cfg = activation_str2dict_mapping(act)
        if 'norm' == 'bn' or 'norm' == 'in':
            norm += '1d'
        norm_cfg = norm_str2dict_mapping(norm)
        drop_cfg = drop_float2cfg_mapping(drop, 'Dropout')
        super(LinearModules, self).__init__(channels_list, kernel_size, stride, padding, dilation, groups, bias,
                                            conv_cfg, norm_cfg, act_cfg, drop_cfg, inplace, with_spectral_norm,
                                            padding_mode, order, last_lin
                                            )
