'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/ops/__init__.py
'''
from .encoding import Encoding
from .wrappers import Upsample, resize

__all__ = ['Upsample', 'resize', 'Encoding']
