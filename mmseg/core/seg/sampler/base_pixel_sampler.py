'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/seg/sampler/base_pixel_smaplesr.py
'''
from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        """Placeholder for sample function."""
