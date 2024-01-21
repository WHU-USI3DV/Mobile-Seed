'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/backbone/__init__.py
'''


from .afformer_for_MS import AFFormer_for_MS_base,AFFormer_for_MS_small,AFFormer_for_MS_tiny
from .afformer import afformer_base,afformer_small,afformer_tiny

all = [
'afformer_base',
'afformer_small',
'afformer_tiny',
'AFFormer_for_MS_base',
'AFFormer_for_MS_small',
'AFFormer_for_MS_tiny',
]