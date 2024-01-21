'''
# File: MS_head.py
# Author: Youqi Liao
# Affiliate: Wuhan University
# Date: Jan 21, 2024
# Description: Head for Mobile-Seed (AFD included)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import einsum


from mmseg.ops import resize
from ..builder import HEADS
from .aff_head import CLS,BaseDecodeHead
    

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        # self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
        #                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        """Forward function."""
        atten = torch.mean(x, dim=(2, 3), keepdim=True) # + torch.amax(x,dim = (2,3),keepdim=True)
        atten = self.conv_1x1(atten)
        return  x,atten
    
class AFD(nn.Module):
    "Active fusion decoder"
    def __init__(self, s_channels,c_channels, conv_cfg, norm_cfg, act_cfg, h=8):
        super(AFD, self).__init__()
        self.s_channels = s_channels
        self.c_channels = c_channels
        self.h = h
        self.scale = h ** - 0.5
        self.spatial_att = ChannelAtt(s_channels, s_channels, conv_cfg, norm_cfg, act_cfg)
        self.context_att = ChannelAtt(c_channels, c_channels, conv_cfg, norm_cfg, act_cfg)
        self.qkv = nn.Linear(s_channels + c_channels,(s_channels + c_channels) * 3,bias = False)
        self.proj = nn.Linear(s_channels + c_channels, s_channels + c_channels)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b = s_att.shape[0] # h = 1, w = 1
        sc_att = torch.cat([s_att,c_att],1).view(b,-1) # [B,2C]
        qkv = self.qkv(sc_att).reshape(b,1,3,self.h, (self.c_channels + self.s_channels) // self.h).permute(2,0,3,1,4) # [B,2C] -> [B,6C] -> [B,1,3,h,2C // h] -> [3,B,h,1,2C // h]
        q,k,v = qkv[0],qkv[1],qkv[2] # [B,h,1,2C // h]
        k_softmax = k.softmax(dim = 1) # channel-wise softmax operation
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v) # [B,h,2C // h ,2C // h]
        fuse_weight = self.scale * einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v) # [B,h,1,2C // h]
        fuse_weight = fuse_weight.transpose(1,2).reshape(b,-1) # [B,C]
        fuse_weight = self.proj(fuse_weight)
        fuse_weight = self.proj_drop(fuse_weight)
        fuse_weight = fuse_weight.reshape(b,-1,1,1) # [B,C,1,1]
        fuse_s,fuse_c = fuse_weight[:,:self.s_channels],fuse_weight[:,-self.c_channels:]
        out = (1 + fuse_s) * s_feat + (1 + fuse_c) * c_feat
        return s_feat, c_feat, out

    





@HEADS.register_module()
class BoundaryHead(BaseDecodeHead):
    def __init__(self,bound_channels,bound_ratio,**kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.bound_ratio = bound_ratio

        self.conv_seg = nn.Conv2d(self.channels,self.num_classes,1)
        self.align0 = ConvModule(self.in_channels[0],out_channels=bound_channels[0],kernel_size=3,stride=1,padding=1,conv_cfg=self.conv_cfg,norm_cfg=dict(type='GN', num_groups=16, requires_grad=True))
        self.align1 = ConvModule(self.in_channels[1],out_channels=bound_channels[1],kernel_size=3,stride=1,padding=1,conv_cfg=self.conv_cfg,norm_cfg=dict(type='GN', num_groups=16, requires_grad=True))
        self.align2 = ConvModule(self.in_channels[2],out_channels=bound_channels[2],kernel_size=3,stride=1,padding=1,conv_cfg=self.conv_cfg,norm_cfg=dict(type='GN', num_groups=16, requires_grad=True))
        self.align3 = ConvModule(self.in_channels[3],out_channels=bound_channels[3],kernel_size=3,stride=1,padding=1,conv_cfg=self.conv_cfg,norm_cfg=dict(type='GN', num_groups=16, requires_grad=True))

    
    def forward(self, seg_feat,img_shape,infer = False):
        seg_feat = self._transform_inputs(seg_feat)
        bound_shape = tuple(i // self.bound_ratio for i in img_shape[:2])
        bound_feat0 = resize(self.align0(seg_feat[0]),size = bound_shape,mode = "bilinear")
        bound_feat1 = resize(self.align1(seg_feat[1]),size = bound_shape,mode = 'bilinear')
        bound_feat2 = resize(self.align2(seg_feat[2]),size = bound_shape,mode = 'bilinear')
        bound_feat3 = resize(self.align3(seg_feat[3]),size = bound_shape,mode = 'bilinear')
        bound_feat = torch.cat([bound_feat0,bound_feat1,bound_feat2,bound_feat3],1)
        bound_logit = self.conv_seg(bound_feat)

        # edge_logit = self.conv_seg(edge_feat)

        return bound_feat,bound_logit

    def forward_test(self, seg_feat, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        bound_feat, bound_logit  =self.forward(seg_feat,img_metas[0]['pad_shape'],infer = False)
        return bound_feat,bound_logit
    
    def losses(self, bound_logit, sebound_label):
        loss = dict()

        # edge_logit = rearrange(edge_logit,'(B Ch Cw) C h w -> B C (Ch h) (Cw w)',Ch=4,Cw=4)
        bound_logit = resize(bound_logit,sebound_label.shape[2:],mode = 'bilinear')

        # edge_feat = resize(edge_feat,segedge_label.shape[2:],mode = 'bilinear')

        bound_label = sebound_label.sum(axis = 1,keepdim=True).float()
        bound_label[bound_label >= 255] = 255.0 # ignore
        bound_label[(bound_label > 0) * (bound_label < 255)] = 1.0

        loss['loss_be'] = self.loss_decode(bound_logit,bound_label)
        # loss['loss_be_int'] = sum(self.loss_decode(edge_feat[:,i : i + 1],edge_label) for i in range(edge_feat.shape[1]))

        return loss

    def forward_train(self, seg_feat,img_refine, img_metas, gt_semantic_segedge,train_cfg):
        bound_feat,bound_logit = self.forward(seg_feat,img_metas[0]['pad_shape']) # imgs in a mini-batch should share the same shape
        losses = self.losses(bound_logit, gt_semantic_segedge)
        return bound_feat,bound_logit,losses

@HEADS.register_module()
class RefineHead(BaseDecodeHead):
    def __init__(self,
                 fuse_channel,
                 **kwargs):
        super().__init__(input_transform='multiple_select',**kwargs)
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.classifer = nn.Conv2d(fuse_channel, self.num_classes,1)
        
        # self.up =  up_block([self.channels,self.channels // 4],[self.channels // 4,self.channels // 4],groups = 32)
        self.boundary_filter_weight = torch.zeros((25,1,5,5),dtype = torch.float32) # frozen
        
        self.boundary_filter_weight[:,:,2,2] = 1.0 # center pixel
        self.boundary_filter_weight[2,:,0,2] = -1.0
        self.boundary_filter_weight[6,0,1,1] = -1.0
        self.boundary_filter_weight[7,0,1,2] = -1.0
        self.boundary_filter_weight[8,0,1,3] = -1.0
        self.boundary_filter_weight[10,0,2,0] = -1.0
        self.boundary_filter_weight[11,0,2,1] = -1.0
        self.boundary_filter_weight[13,0,2,3] = -1.0
        self.boundary_filter_weight[14,0,2,4] = -1.0
        self.boundary_filter_weight[16,0,3,1] = -1.0
        self.boundary_filter_weight[17,0,3,2] = -1.0
        self.boundary_filter_weight[18,0,3,3] = -1.0
        self.boundary_filter_weight[22,0,4,2] = -1.0
        
        # for i in range(5):
        #     for j in range(5):
        #        self.edge_filter_weight[5* i + j,:,i,j] -= 1.0
        self.boundary_filter = nn.Conv2d(1,25,5,1,2,bias = False,padding_mode='reflect') # 'reflect' padding to refrain conflict
        self.boundary_filter.weight.data = self.boundary_filter_weight
        self.boundary_filter.weight.requires_grad = False
        
        # semantic and boundary feature fusion module 
        self.bs_fusion = AFD(fuse_channel,fuse_channel,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg,h = 8)

        self.align_fuse = ConvModule(
             self.in_channels[-1],
             fuse_channel,
             1,
             conv_cfg=self.conv_cfg,
             norm_cfg=self.norm_cfg,
             act_cfg=self.act_cfg)

    def forward(self,seg_feat,bound_feat,img_shape,infer = False):
        """Forward function."""
        # B,C,H,W = edge_feat.shape
        seg_feat = self._transform_inputs(seg_feat)[0]
        seg_feat_ori = self.squeeze(seg_feat)
        # weight_feat = self.squeeze(seg_feat)
        # if seg_feat.shape[2] != edge_feat.shape[2]:
        #     seg_feat = resize(seg_feat,size = edge_feat.shape[2:],mode = 'bilinear')
        seg_logit = None
        if infer == False:
            seg_logit = self.cls_seg(seg_feat_ori)

        seg_feat_fuse = resize(self.align_fuse(seg_feat),size = bound_feat.shape[2:],mode = "bilinear")
        # seg_feat_fuse = self.align_fuse(seg_feat)
        # seg_feat_fuse = self.up(seg_feat)
        # seg_feat_fuse = torch.cat([seg_feat_fuse,edge_feat],1)
        _,_,seg_feat_fuse = self.bs_fusion(seg_feat_fuse,bound_feat)

        seg_logit_fuse = self.classifer(seg_feat_fuse)


        return seg_logit,seg_logit_fuse
    
    def forward_train(self, seg_feat,bound_feat, bound_logit, img_metas, gt_semantic_seg, gt_semantic_segedge, train_cfg):
        losses = dict()

        seg_logit,segfine_logit = self.forward(seg_feat,bound_feat,gt_semantic_seg.shape[2:])
        
        bound_label = gt_semantic_segedge.sum(axis = 1,keepdim=True).float()
        bound_label[bound_label >= 255] = 255.0 # ignore
        bound_label[(bound_label > 0) * (bound_label < 255)] = 1.0
        
        loss_decode = self.losses(seg_logit,gt_semantic_seg)
        losses.update(loss_decode)
        
        loss_decodefine = self.losses(segfine_logit,gt_semantic_seg,acc_name='acc_sefine',loss_name='loss_sefine',loss_weight = 1.0)
        losses.update(loss_decodefine)

        sebound_prob = torch.zeros_like(segfine_logit)
        segfine_prob = segfine_logit.softmax(dim = 1)
        for i in range(sebound_prob.shape[1]):
            sebound_prob[:,i] = self.boundary_filter(segfine_prob[:,i:i+1]).abs().max(dim = 1)[0]
        sebound_prob = resize(sebound_prob,size = bound_label.shape[2:],mode = 'bilinear')
        loss_sebound = dict()
        loss_sebound['loss_sebound'] = 1.0 * F.l1_loss(sebound_prob,gt_semantic_segedge,reduction='mean')
        losses.update(loss_sebound)
        
        
        bound_logit = resize(bound_logit,size = bound_label.shape[2:],mode = 'bilinear')
        # segfine_logit = resize(segfine_logit,size = edge_label.shape[2:],mode = 'bilinear')

        # regularization loss
        gt_semantic_seg_hard = torch.clone(gt_semantic_seg)
        bound_hard_mask = torch.logical_or(bound_logit > 0.8,bound_label == 1.0)
        gt_semantic_seg_hard[~bound_hard_mask] = 255.0 # ignore mask
        loss_decodehard = self.losses(segfine_logit,gt_semantic_seg_hard,acc_name = "acc_hard",loss_name = "loss_hard",loss_weight = 1.0)
        losses.update(loss_decodehard)
        
        return losses
    
    def forward_test(self, seg_feat,bound_feat, img_metas, test_cfg):
        # only return segment result in default
        _,segfine_logit = self.forward(seg_feat,bound_feat,None,infer= True)

        return segfine_logit

