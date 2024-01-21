'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/formating.py
'''
import warnings

from .formatting import *

warnings.warn('DeprecationWarning: mmseg.datasets.pipelines.formating will be '
              'deprecated in 2021, please replace it with '
              'mmseg.datasets.pipelines.formatting.')
