import torch.nn as nn
from .encoder.assanet_encoder import ASSANetEncoder
from .decoder.assanet_decoder import ASSANetDecoder
from .segmentation_head import SceneSegHeadPointNet
from pytorch_lightning.core import LightningModule
from .utils.config import config


class ASSANetSeg(LightningModule):
    def __init__(self, in_channels=3, out_channels=20, **kwargs):
        """ASSA-Net implementation for paper:
        Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning

        Args:
            cfg (dict): configuration
        """
        super().__init__()
        self.save_hyperparameters()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)
        self.out_channels = out_channels
        config.load(self.cfg, recursive=True)
        self.encoder = ASSANetEncoder(config, in_channels)
        self.decoder = ASSANetDecoder(config)
        self.head = SceneSegHeadPointNet(num_classes=out_channels, in_channels=config.model.fp_mlps[0][0])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch, return_feats=False):
        features = batch['feats']
        xyz = batch['pts']
        l_xyz, l_features = self.encoder(xyz, features)
        out_f = self.decoder(l_xyz, l_features)
        out = self.head(out_f)
        # print(out.shape)
        if return_feats:
            return out.transpose(1,2).contiguous().reshape(-1, self.out_channels), None
        return out.transpose(1,2).contiguous().reshape(-1, self.out_channels)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ASSANetSeg")
        parser.add_argument("--cfg", type=str, default='/home/zarzarj/git/MinkowskiEngine/examples/ASSA/cfgs/s3dis/assanet.yaml')
        return parent_parser

    def convert_sync_batchnorm(self):
        return
