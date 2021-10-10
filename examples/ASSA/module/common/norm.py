import inspect

import torch.nn as nn

from .misc import is_tuple_of
from ..common.registry import NORM_LAYERS

NORM_LAYERS.register_module('BN', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN1d', module=nn.BatchNorm1d)
NORM_LAYERS.register_module('BN2d', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN3d', module=nn.BatchNorm3d)
NORM_LAYERS.register_module('SyncBN', module=nn.SyncBatchNorm)
NORM_LAYERS.register_module('GN', module=nn.GroupNorm)
NORM_LAYERS.register_module('LN', module=nn.LayerNorm)
NORM_LAYERS.register_module('IN', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN1d', module=nn.InstanceNorm1d)
NORM_LAYERS.register_module('IN2d', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN3d', module=nn.InstanceNorm3d)



NORM_MAPPING = {
    'bn': 'BN',
    'bn1d': 'BN1d',
    'bn2d': 'BN2d',
    'bn3d': 'BN3d',
    'syncbn': 'SyncBN',
    'gn': 'GN',
    'ln': 'LN',
    'in': 'IN',
    'in1d': 'IN1d',
    'in2d': 'IN2d',
    'in3d': 'IN3d',
}


def norm_str2dict_mapping(norm='bn'):
    """ Map the input args of act (string) into the dictionary for building activation layer.
    Args:
        act (string): lowercase string for activation layer

    Returns:

    """
    assert isinstance(norm, str) or norm is None
    if norm is not None:
        norm = {'type': NORM_MAPPING[norm.lower()]}
    return norm


def infer_abbr(class_type):
    """Infer abbreviation from the class name.
    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.
    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".
    Args:
        class_type (type): The norm layer type.
    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, nn.modules.instancenorm._InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, nn.modules.batchnorm._BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm'


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.
    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
    Returns:
        nn.Module: created norm layer.

    Example:
        >>> # ---------------- Activation
        >>> n_feats = 512
        >>> n_points = 128
        >>> n_batch = 4
        >>> device = torch.device('cuda')
        >>> feats = torch.rand((n_batch, n_feats, n_points), dtype=torch.float).to(device) - 0.5
        >>> print(f"before operation, "
        >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")
        >>> norm = build_norm_layer({'type': 'BN2d', 'eps': 1e-5}, num_features=n_feats).to(device)
        >>> feats = norm(feats)
        >>> print(f"after operation, "
        >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")

    """
    if not isinstance(cfg, dict):
        assert isinstance(cfg, str)
        cfg = norm_str2dict_mapping(cfg)
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


def is_norm(layer, exclude=None):
    """Check if a layer is a normalization layer.
    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.
    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm,
                      nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)

