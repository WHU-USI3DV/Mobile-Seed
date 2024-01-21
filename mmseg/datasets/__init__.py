'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/__inti__.py
'''

from .cityscapes import CityscapesDataset,CityscapesDataset_boundary
from .camvid import CamVidDataset,CamVidDataset_boundary
from .pascal_context import PascalContextDataset,PascalContextDataset59,PascalContextDataset_boundary,PascalContextDataset59_boundary
from .custom import CustomDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset


__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES','MultiImageMixDataset', 'CityscapesDataset', 'CityscapesDataset_boundary', 
    'CamVidDataset','CamVidDataset_boundary','PascalContextDataset','PascalContextDataset59', 'PascalContextDataset_boundary',
    'PascalContextDataset59_boundary'
]
