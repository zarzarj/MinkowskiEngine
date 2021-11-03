# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from examples.resnet import ResNetBase

class MinkUNetTwoStreamBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels=3, out_channels=20, bn_momentum=0.1, D=3):
        self.bn_momentum=bn_momentum
        ResNetBase.__init__(self, in_channels, out_channels, D)
        

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes, momentum=self.bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=self.bn_momentum)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=self.bn_momentum)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=self.bn_momentum)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=self.bn_momentum)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=self.bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=self.bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=self.bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=self.bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)


        if self.gnn_model == 'rgat':
            conv = GATConv(in_channels, self.hidden_channels // self.heads, self.heads,
                              add_self_loops=False)
        self.gnn_convs = []
        for i in range(self.num_layers-1):
            self.gnn_convs.append(copy.deepcopy(conv))
            self.gnn_convs[-1].reset_parameters()
            
        self.gnn_skips = ModuleList()
        self.gnn_skips.append(Linear(in_channels, self.hidden_channels))
        for _ in range(self.num_layers - 1):
            self.gnn_skips.append(Linear(self.hidden_channels, self.hidden_channels))

    def forward(self, in_dict):
        adj = in_dict['adj']
        in_field = ME.TensorField(
            features=in_dict['feats'],
            coordinates=in_dict['coords'],
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            # minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=self.device,
        )
        # print(in_field)
        x = in_field.sparse()
        out = self.conv0p1s1(x)
        gnn_out = self.convs[0](in_dict['feats'], adj['p1'])
        out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.bn0(out)
        out_p1 = self.relu(out)
        

        out = self.conv1p1s2(out_p1)
        gnn_out = self.convs[1](gnn_out, adj['p2'])
        out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.bn1(out)
        out = self.relu(out)

        out_b1p2 = self.block1(out)
        # gnn_out = self.convs[0](out._F, adj[0])
        # out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.conv2p2s2(out_b1p2)
        gnn_out = self.convs[2](gnn_out, adj['p4'])
        out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.bn2(out)
        out = self.relu(out)

        out_b2p4 = self.block2(out)
        # gnn_out = self.convs[0](in_dict['feats'], adj[0])
        # out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.conv3p4s2(out_b2p4)
        gnn_out = self.convs[3](gnn_out, adj['p8'])
        out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.bn3(out)
        out = self.relu(out)

        out_b3p8 = self.block3(out)
        # gnn_out = self.convs[0](in_dict['feats'], adj[0])
        # out._F = torch.cat([out._F, gnn_out], axis=-1)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        gnn_out = self.convs[4](gnn_out, adj['p16'])
        out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        # gnn_out = self.convs[cur_gnn_layer](out._F, adj['p1'])
        # out._F = torch.cat([out._F, gnn_out], axis=-1)

        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)
        out = self.final(out)

        # if in_dict['rand_shift'] is not None:
        #     coords = []
        #     for i in range(len(in_dict['rand_shift'])):
        #         coords.append( out.coordinates_at(i) - in_dict['rand_shift'][i])
        #     feats = out.decomposed_features
        # else:
        #     coords, feats = out.decomposed_coordinates_and_features
        feats = out.slice(in_field).F
        # feats = torch.cat(feats, axis=0)
        return feats

    @staticmethod
    def add_argparse_args(parent_parser):
        parser.add_argument("--hidden_channels", type=int, default=128)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--dropout", type=int, default=0.0)
        parser.add_argument("--bn_momentum", type=int, default=0.1)
        parser.add_argument("--norm", type=str, default='batch', choices=['batch', 'layer', 'instance', 'none'])
        parser.add_argument("--track_running_stats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--reversible", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--model", type=str, default='rgat', choices=['rgat'])
        return parent_parser

    def convert_sync_batchnorm(self):
        self = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self)


class MinkUNetTwoStream14(MinkUNetTwoStreamBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNetTwoStream34(MinkUNetTwoStreamBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class MinkUNetTwoStream34C(MinkUNetTwoStream34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
