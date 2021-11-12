from typing import Any, Optional, List, NamedTuple
from examples.str2bool import str2bool
from examples.BaseLightningPointNet import BasePointNetLightning, BaseWholeScene, BaseChunked
from examples.S3DISBase import S3DISBase

class S3DISPointNet(S3DISBase, BasePointNetLightning):
    def __init__(self, **kwargs):
        BasePointNetLightning.__init__(self, **kwargs)
        S3DISBase.__init__(self, **kwargs)
        self.kwargs = copy.deepcopy(kwargs)
        self.whole_scene_dataset = S3DISWholeScene
        self.chunked_scene_dataset = S3DISChunked

    def setup(self, stage: Optional[str] = None):
        S3DISBase.setup(self, stage)
        BasePointNetLightning.setup(self, stage)

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