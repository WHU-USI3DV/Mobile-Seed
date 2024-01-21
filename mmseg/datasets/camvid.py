'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/cityscapes.py
'''
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamVidDataset(CustomDataset):
    """CamVid dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png', ``seg_map_suffix`` is
    fixed to '_trainIds.png' and ``sebound_map_suffix`` is fixed to '_edge.png' for CamVid dataset.
    """

    CLASSES = ('Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence', 'Pedestrian',
               'Road', 'Sidewalk', 'SignSymbol', 'Sky','Tree')

    PALETTE = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_trainIds.png',
                 **kwargs):
        super(CamVidDataset, self).__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            **kwargs)
        # self.segedge_map_suffix = segedge_map_suffix

        # reload annotations (segedge map)
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix,self.segedge_map_suffix, self.split)


@DATASETS.register_module()
class CamVidDataset_boundary(CustomDataset):
    """CamVid dataset with edge label for Mobile-Seed.

    The ``img_suffix`` is fixed to '_leftImg8bit.png', ``seg_map_suffix`` is
    fixed to '_trainIds.png' and ``sebound_map_suffix`` is fixed to '_edge.png' for CamVid dataset.
    """

    CLASSES = ('Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence', 'Pedestrian',
               'Road', 'Sidewalk', 'SignSymbol', 'Sky','Tree')

    PALETTE = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_trainIds.png',
                 sebound_map_suffix = '_edge.png',
                 **kwargs):
        super(CamVidDataset_boundary, self).__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            sebound_map_suffix=sebound_map_suffix,
            **kwargs)
        # self.segedge_map_suffix = segedge_map_suffix

        # reload annotations (segedge map)
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #           