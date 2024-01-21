'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/segmentors/encoder_decoder.py
'''
import torch
import time
from torch import nn
#from ..decode_heads.lpls_utils import Lap_Pyramid_Conv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.runner import auto_fp16
import torch.nn.functional as F
from einops import rearrange

@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder):
    """Cascade Encoder Decoder segmentors for semantic segmentation and boundary detection dual-task learning.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 down_ratio,
                 backbone,
                 decode_head,
                 refine_input_ratio=1.,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 is_frequency=False,
                 pretrained=None,
                 init_cfg=None):
        self.is_frequency = is_frequency
        self.down_scale = down_ratio
        self.refine_input_ratio = refine_input_ratio
        super(EncoderDecoderRefine, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)

        self.boundary_head = builder.build_head(decode_head[0])
        self.decode_head = builder.build_head(decode_head[1])
        # print(self.decode_head)
        # print(self.refine_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
    
    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # img_dw = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])

        # if self.refine_input_ratio == 1.:
        #     img_refine = img
        # elif self.refine_input_ratio < 1.:
        #     img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])

        # img_dw = rearrange(img_dw,'B C (Ch h) (Cw w) -> (B Ch Cw) C h w',Ch=4,Cw=4)
        seg_feat = self.extract_feat(img)
        bound_feat,bound_logit = self.boundary_head.forward_test(seg_feat, img_metas, self.test_cfg)
        seg_logit = self.decode_head.forward_test(seg_feat,bound_feat,img_metas, self.test_cfg)

        return seg_logit,bound_logit
    
    def forward_train(self, img, img_metas, gt_semantic_seg,gt_semantic_sebound):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # crop the channel dimension of gt_semantic_sebound
        gt_semantic_sebound = gt_semantic_sebound.permute(0,3,1,2).float()
        gt_semantic_sebound = gt_semantic_sebound[:,:self.num_classes,...]

        # img_dw = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])

        # if self.refine_input_ratio == 1.:
        #     img_refine = img
        # elif self.refine_input_ratio < 1.:
        #     img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        
        # img_dw = rearrange(img_dw,'B C (Ch h) (Cw w) -> (B Ch Cw) C h w',Ch=4,Cw=4)

        seg_feat = self.extract_feat(img)

        losses = dict()
        loss_decode = self._decode_head_forward_train(seg_feat,img,img_metas,gt_semantic_seg,gt_semantic_sebound)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                seg_feat, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO: 搭建refine的head
    def _decode_head_forward_train(self, seg_feat, img, img_metas, gt_semantic_seg,gt_semantic_sebound):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        
        bound_feat,bound_logit, loss_bound  = self.boundary_head.forward_train(
            seg_feat,img, img_metas, gt_semantic_sebound, self.train_cfg)
        losses.update(loss_bound)

        loss_decode = self.decode_head.forward_train(seg_feat, bound_feat, bound_logit, img_metas, gt_semantic_seg, gt_semantic_sebound, self.train_cfg)
        losses.update(loss_decode)

        return losses

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        # preds_edge = img.new_zeros((batch_size, self.num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit,_ = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                # preds_edge += F.pad(crop_edge_logit,
                #                (int(x1), int(preds_edge.shape[3] - x2), int(y1),
                #                 int(preds_edge.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        # preds_edge = preds_edge / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            
            # preds_edge = resize(
            #     preds_edge,
            #     size=img_meta[0]['ori_shape'][:2],
            #     mode='bilinear',
            #     align_corners=self.align_corners,
            #     warning=False)
            
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit,bound_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            
            bound_logit = resize(
                 bound_logit,
                 size=size,
                 mode='bilinear',
                 align_corners=self.align_corners,
                 warning=False)

        return seg_logit,bound_logit
    
    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit,bound_logit = self.whole_inference(img, img_meta, rescale)
        # output = F.softmax(seg_logit, dim=1)
        # output_edge = segedge_logit
        # output_edge = F.softmax(segedge_logit,dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return seg_logit,bound_logit
    
    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit,bound_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        bound_pred = bound_logit.sigmoid().squeeze(1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        bound_pred = bound_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        bound_pred = list(bound_pred)
        return seg_pred,bound_pred
