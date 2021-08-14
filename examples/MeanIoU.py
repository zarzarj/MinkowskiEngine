from torchmetrics import Metric
import torch

class MeanIoU(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)

        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        for cl in range(self.num_classes):
            gt_mask = (target == cl)
            pred_mask = (preds == cl)
            self.intersection[cl] += torch.sum(torch.logical_and(pred_mask, gt_mask))
            self.union[cl] += torch.sum(torch.logical_or(pred_mask, gt_mask))

    def compute(self):
        ious = self.intersection.float() / self.union
        ious = ious[~ious.isnan()]
        return torch.mean(ious)