'''
This file is modified from:
https://github.com/dongbo811/AFFormer/blob/main/mmseg/models/decode_heads/aff_head.py
'''
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *

@HEADS.register_module()
class CLS(BaseDecodeHead):
    def __init__(self,
                 aff_channels=512,
                 aff_kwargs=dict(),
                 **kwargs):
        super(CLS, self).__init__(
                input_transform='multiple_select', **kwargs)
        self.aff_channels = aff_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        

        self.align = ConvModule(
             self.aff_channels,
             self.channels,
             1,
             conv_cfg=self.conv_cfg,
             norm_cfg=self.norm_cfg,
             act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)[0]

        x = self.squeeze(inputs)
            
        output = self.cls_seg(x)
        return output

@HEADS.register_module()
class CLS_edge(BaseDecodeHead):
    def __init__(self,
                 aff_channels=512,
                 aff_kwargs=dict(),
                 **kwargs):
        super(CLS_edge, self).__init__(
                input_transform='multiple_select', **kwargs)
        self.aff_channels = aff_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        

        self.align = ConvModule(
             self.aff_channels,
             self.channels,
             1,
             conv_cfg=self.conv_cfg,
             norm_cfg=self.norm_cfg,
             act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)[0]

        x = self.squeeze(inputs)
            
        output = self.cls_seg(x)
        return output
    
    def losses(self, edge_logit, segedge_label):
        loss = dict()

        # edge_logit = rearrange(edge_logit,'(B Ch Cw) C h w -> B C (Ch h) (Cw w)',Ch=4,Cw=4)
        edge_logit = resize(edge_logit,segedge_label.shape[2:],mode = 'bilinear')

        # edge_feat = resize(edge_feat,segedge_label.shape[2:],mode = 'bilinear')

        edge_label = segedge_label.sum(axis = 1,keepdim=True).float()
        edge_label[edge_label >= 255] = 255.0 # ignore
        edge_label[(edge_label > 0) * (edge_label < 255)] = 1.0

        loss['loss_be'] = self.loss_decode(edge_logit,edge_label)
        # loss['loss_be_int'] = sum(self.loss_decode(edge_feat[:,i : i + 1],edge_label) for i in range(edge_feat.shape[1]))

        return loss
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, gt_semantic_segedge, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logit = self.forward(inputs)

        losses = self.losses(seg_logit, gt_semantic_segedge)
        return losses