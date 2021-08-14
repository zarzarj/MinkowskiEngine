import torch
import torch.nn as nn
from torch.optim import SGD

from pytorch_lightning.core import LightningModule
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C


class MinkowskiSegmentationModule(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        self.criterion = nn.CrossEntropyLoss()
        self.model = MinkUNet34C(self.in_channels, self.out_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        coords, input, target = batch
        input = input.float()
        # For some networks, making the network invariant to even, odd coords is important
        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
        sinput = ME.SparseTensor(input, coords)
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return self.criterion(self(sinput).F, target.long())

    def validation_step(self, batch, batch_idx):
        coords, input, target = batch
        input = input.float()
        # For some networks, making the network invariant to even, odd coords is important
        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
        sinput = ME.SparseTensor(input, coords)
        return self.criterion(self(sinput).F, target.long())

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MinkSegModel")
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--out_channels", type=int, default=20)
        parser.add_argument("--optimizer_name", type=str, default='SGD')
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parent_parser