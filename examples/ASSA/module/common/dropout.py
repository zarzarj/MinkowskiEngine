import torch.nn as nn

from .registry import build_from_cfg
from ..common.registry import DROPOUT_LAYERS

for module in [
        nn.Dropout, nn.Dropout2d, nn.Dropout3d
]:
    DROPOUT_LAYERS.register_module(module=module)


def build_dropout_layer(cfg, inplace=False):
    """Build dropout layer.
    Args:
        cfg (dict): The dropout layer config, which should contain:
            - type (str): Layer type. (Dropout2d, Dropout3d)
        inplace (bool): Set to use inplace operation.
    Returns:
        nn.Module: Created dropout layer.

    Example:
            >>> # ---------------- Dropout
            >>> n_feats = 512
            >>> n_points = 128
            >>> n_batch = 4
            >>> device = torch.device('cuda')
            >>> feats = torch.rand((n_batch, n_feats, n_points), dtype=torch.float).to(device) - 0.5
            >>> print(f"before operation, "
            >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")

            >>> # build activation
            >>> drop_cfg = {'type': 'Dropout2d'}
            >>> dropout = build_dropout_layer(drop_cfg).to(device)

            >>> # test activation
            >>> feats = dropout(feats)
            >>> print(f"after operation, "
            >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")

    """
    assert cfg['p'] > 0.
    cfg.setdefault('inplace', inplace)
    return build_from_cfg(cfg, DROPOUT_LAYERS)


def drop_float2cfg_mapping(drop_prob=0., type='Dropout2d'):
    """ Map input drop_prob (float) into the drop_dict
    Args:
        drop_prob (float): dropout probability

    Returns:
        drop_cfg
    """
    assert isinstance(drop_prob, float)
    if drop_prob > 0.:
        drop_cfg = {'type': type, 'p': drop_prob}
    else:
        drop_cfg = None
    return drop_cfg


