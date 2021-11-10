from typing import Any, Optional, List, NamedTuple
from examples.str2bool import str2bool
from examples.BaseLightningPointNet import BasePointNetLightning, BaseWholeScene, BaseChunked
from examples.S3DISBase import S3DISBase

class S3DISPointNet(S3DISBase, BasePointNetLightning):
    def __init__(self, **kwargs):
        BasePointNetLightning.__init__(self, **kwargs)
        S3DISBase.__init__(self, **kwargs)

    def setup(self, stage: Optional[str] = None):
        S3DISBase.setup(self, stage)
        if self.use_whole_scene:
            self.train_dataset = S3DISWholeScene(phase="train", scene_list=self.train_files, **self.kwargs)
            self.val_dataset = S3DISWholeScene(phase="val", scene_list=self.val_files, **self.kwargs)
        else:
            self.train_dataset = S3DISChunked(phase="train", scene_list=self.train_files, **self.kwargs)
            self.val_dataset = S3DISChunked(phase="val", scene_list=self.val_files, **self.kwargs)
        # self.val_dataset.generate_chunks()

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BasePointNetLightning.add_argparse_args(parent_parser)
        parent_parser = S3DISBase.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("S3DISPointNet")
        return parent_parser


class S3DISWholeScene(S3DISBase, BaseWholeScene):
    def __init__(self, **kwargs):
        BaseWholeScene.__init__(self, **kwargs)
        S3DISBase.__init__(self, **kwargs)

class S3DISChunked(S3DISBase, BaseChunked):
    def __init__(self, **kwargs):
        BaseChunked.__init__(self, **kwargs)
        S3DISBase.__init__(self, **kwargs)