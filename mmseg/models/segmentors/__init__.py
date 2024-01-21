'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/segmentors/__init__.py
'''
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder,EncoderDecoder_edge
from .encoder_decoder_refine import EncoderDecoderRefine

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder','EncoderDecoder_edge','EncoderDecoderRefine']
