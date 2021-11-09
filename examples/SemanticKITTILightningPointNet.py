from typing import Any, Optional, List, NamedTuple
from examples.str2bool import str2bool
from examples.BaseLightningPointNet import BasePointNetLightning, BaseWholeScene, BaseChunked

class SemanticKITTIPointNet(SemanticKITTIBase, BasePointNetLightning):
    def __init__(self, **kwargs):
        BasePointNetLightning.__init__(self, **kwargs)
        SemanticKITTIBase.__init__(self, **kwargs)

    def setup(self, stage: Optional[str] = None):
        SemanticKITTIBase.setup(self, stage)
        if self.use_whole_scene:
            self.train_dataset = ScannetWholeScene(phase="train", scene_list=self.train_scans, **self.kwargs)
            self.val_dataset = ScannetWholeScene(phase="val", scene_list=self.val_scans, **self.kwargs)
        else:
            self.train_dataset = ScannetChunked(phase="train", scene_list=self.train_scans, **self.kwargs)
            self.val_dataset = ScannetChunked(phase="val", scene_list=self.val_scans, **self.kwargs)
        # self.val_dataset.generate_chunks()

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePointNetLightning.add_argparse_args(parent_parser)
        parent_parser = SemanticKITTIBase.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("SemanticKITTIPointNet")
        return parent_parser


class SemanticKITTIWholeScene(SemanticKITTIBase, BaseWholeScene):
    def __init__(self, **kwargs):
        BaseWholeScene.__init__(self, **kwargs)
        SemanticKITTIBase.__init__(self, **kwargs)

class SemanticKITTIChunked(SemanticKITTIBase, BaseChunked):
    def __init__(self, **kwargs):
        BaseChunked.__init__(self, **kwargs)
        SemanticKITTIBase.__init__(self, **kwargs)