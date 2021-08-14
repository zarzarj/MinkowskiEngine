from torchmetrics import Metric
import torch

class MeanAccuracy(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)

        self.add_state("correct", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        for i in range(self.num_classes):
            self.correct[i] += torch.sum(torch.logical_and(preds == target, target == i))
            self.total[i] += torch.sum(target == i)

    def compute(self):
        accs = self.correct.float() / self.total
        accs = accs[~accs.isnan()]
        # print(self.correct)
        return torch.mean(accs)