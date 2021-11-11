import torch
from .gcn_lib.dense import BasicConv, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph
from torch.nn import Sequential as Seq


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=20, **kwargs):
        super(DenseDeepGCN, self).__init__()
        # self.save_hyperparameters()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)
        channels = self.n_filters
        k = self.k
        act = self.act
        norm = self.norm
        bias = self.bias
        epsilon = self.epsilon
        stochastic = self.stochastic
        conv = self.conv
        c_growth = channels
        self.out_channels=out_channels

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels+3, channels, conv, act, norm, bias)

        if self.block.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock2d(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif self.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon)
                                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims+1024, 512], act, norm, bias),
                                BasicConv([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=self.dropout),
                                BasicConv([256, out_channels], None, None, bias)])

    def forward(self, batch, return_feats=False):
        # print(batch['coords'].shape, batch['feats'].shape)
        inputs = torch.cat([batch['pts'].transpose(1,2), batch['feats']], axis=1).unsqueeze(-1)
        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        out = self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)
        # print(out.shape)
        if return_feats:
            return out.transpose(1,2).contiguous().reshape(-1, self.out_channels), None
        return out.transpose(1,2).contiguous().reshape(-1, self.out_channels)

    def convert_sync_batchnorm(self):
        return
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DenseDeepGCN")
        parser = parent_parser.add_argument('--k', default=20, type=int, help='neighbor num (default:16)')
        parser = parent_parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
        parser = parent_parser.add_argument('--conv', default='mr', type=str, help='graph conv layer {edge, mr}')
        parser = parent_parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser = parent_parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
        parser = parent_parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
        parser = parent_parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
        parser = parent_parser.add_argument('--n_blocks', default=28, type=int, help='number of basic blocks')
        parser = parent_parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        parser = parent_parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser = parent_parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
        return parent_parser