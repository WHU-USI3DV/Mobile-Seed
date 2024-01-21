'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/seg/sampler/__init__.py
'''
from .base_pixel_sampler import BasePixelSampler
from .ohem_pixel_sampler import OHEMPixelSampler

__all__ = ['BasePixelSampler', 'OHEMPixelSampler']
