import torch
from torch import tensor
from torchmetrics import Metric

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += preds.shape[0]

    def compute(self):
        if self.total == 0:
            return 0
        acc = self.correct.float() / self.total
        return acc