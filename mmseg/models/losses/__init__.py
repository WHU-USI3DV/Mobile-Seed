'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/losses/__init__.py
'''
from .accuracy import Accuracy, accuracy,accuracy_se
from .cross_entropy_loss import (CrossEntropyLoss, ML_BCELoss, L1_loss,NLL_loss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss','ML_BCELoss','L1_loss','NLL_loss','accuracy_se'
]
