'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/__init__.py
'''
from .decode_head import BaseDecodeHead
from .MS_head import RefineHead,BoundaryHead
from .aff_head import CLS
__all__ = ['CLS', 'BaseDecodeHead','RefineHead','BoundaryHead']
