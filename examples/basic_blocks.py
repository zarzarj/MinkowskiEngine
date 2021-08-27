import torch
import torch.nn as nn


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def norm_layer(norm_type, nc, track_running_stats=True):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True, track_running_stats=track_running_stats)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    elif norm == 'none':
        layer = nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

class MLP(nn.Sequential):
    def __init__(self, mlp_channels, act='relu', norm='batch', bias=True, dropout=0.0):
        m = []
        in_channels = mlp_channels[0]
        for hidden_channels in mlp_channels[1:]:
            m.append(nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, hidden_channels))
            if dropout > 0:
                m.append(nn.Dropout2d(dropout, inplace=True))
            in_channels = hidden_channels
        super(MLP, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()