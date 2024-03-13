'''
This file is modified from:
https://github.com/dongbo811/AFFormer/blob/main/tools/afformer.py
'''
import torch
from torch import einsum, nn
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import math
import numpy as np
from einops import rearrange

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import (BaseModule, ModuleList, load_checkpoint)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, size):
        H, W = size
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):


    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride, 
            act_layer=act_layer,
        )

    def forward(self, x):

        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):

    def __init__(self, embed_dim, num_path=4, isPool=False, stage=0):
        super(Patch_Embed_stage, self).__init__()

        if stage == 3:
            self.patch_embeds = nn.ModuleList([
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=4 if (isPool and idx == 0) or (stage > 1 and idx == 1) else 1,
                ) for idx in range(num_path + 1)
            ])
        else:

            self.patch_embeds = nn.ModuleList([
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if (isPool and idx == 0) or (stage > 1 and idx == 1) else 1,
                ) for idx in range(num_path + 1)
            ])

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)

            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        return x

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch =  in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)
        
        return self.relu(bottle)
    
    
class FilterModule(nn.Module):
    def __init__(self, Ch, h, window):

        super().__init__()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.LP = LowPassModule(Ch * h)

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # channel shuffle
        # v_img_mix = v_img[:,shuffle_index,:,:]

        LP = self.LP(v_img)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        HP_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        HP = torch.cat(HP_list, dim=1)

        # reverse to original order
        # HP = HP[:,reverse_index,...]
        # LP = LP[:,reverse_index,...]
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        HP = rearrange(HP, "B (h Ch) H W -> B h (H W) Ch", h=h)
        LP = rearrange(LP, "B (h Ch) H W -> B h (H W) Ch", h=h)

        dynamic_filters = q * HP + LP
        return dynamic_filters


class Frequency_FilterModule(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)) # 3,B,h,N,k
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        
        # shuffle_index = torch.randperm(self.num_heads)
        # reverse_index = shuffle_index.sort()[-1]
        # random_index = shuffle_index.repeat(B,self.num_heads,N)
        # grid_b,grid_h,grid_n = torch.meshgrid([torch.arange(0,B),torch.arange(0,self.num_heads),torch.arange(C // self.num_heads)])
        # mix_v = v[:,shuffle_index,:,:]

        # high/low-pass filters
        crpe = self.crpe(q, v, size=size)
        # crpe = crpe[:,reverse_index,:,:]

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MHCABlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = Frequency_FilterModule(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur, size))
        return x


class MHCAEncoder(nn.Module):
    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            crpe_window={
                3: 2,
                5: 3,
                7: 3
            },
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = FilterModule(Ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)
        self.MHCA_layers = nn.ModuleList([
            MHCABlock(
                dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_list[idx],
                qk_scale=qk_scale,
                shared_cpe=self.cpe,
                shared_crpe=self.crpe,
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class Restore(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = in_features // 2
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            drop_path_list=[],
            id_stage=0,
    ):
        super().__init__()



        self.Restore = Restore(in_features=embed_dim, out_features=embed_dim)
        if id_stage > 0:
            self.aggregate = Conv2d_BN(embed_dim * (num_path),
                                       out_embed_dim,
                                       act_layer=nn.Hardswish)
            self.mhca_blks = nn.ModuleList([
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                ) for _ in range(num_path)
            ])
        else:
            self.aggregate = Conv2d_BN(embed_dim * (num_path),
                                       out_embed_dim,
                                       act_layer=nn.Hardswish)

    def forward(self, inputs, id_stage):

        if id_stage > 0:
            att_outputs = [self.Restore(inputs[0])]
            for x, encoder in zip(inputs[1:], self.mhca_blks):
                # [B, C, H, W] -> [B, N, C]
                _, _, H, W = x.shape

                x = x.flatten(2).transpose(1, 2)
                att_outputs.append(encoder(x, size=(H, W)))

            for i in range(len(att_outputs)):
                if att_outputs[i].shape[2:] != att_outputs[0].shape[2:]:
                    att_outputs[i] = F.interpolate(att_outputs[i], size=att_outputs[0].shape[2:], mode='bilinear',
                                                   align_corners=True)

            out_concat = att_outputs[0] + att_outputs[1]
        else:
            out_concat = self.Restore(inputs[0] + inputs[1])

        out = self.aggregate(out_concat)

        return out


class Cls_head(nn.Module):
    """a linear layer for classification."""

    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # (B, C, H, W) -> (B, C, 1)

        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # Shape : [B, C]
        out = self.cls(x)
        return out


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr



class AFFormer(BaseModule):
    def __init__(
        self,
        img_size=224,
        num_stages=4,
        num_path=[4, 4, 4, 4],
        num_layers=[1, 1, 1, 1],
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        drop_path_rate=0.0,
        in_chans=3,
        num_classes=1000,
        strides=[4, 2, 2, 2],
        pretrained=None, init_cfg=None,
    ):
        super().__init__()
        if isinstance(pretrained, str):
            self.init_cfg = pretrained
        self.num_classes = num_classes
        self.num_stages = num_stages

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=True if idx == 1 else False,
                stage=idx,
            ) for idx in range(self.num_stages)
        ])

        self.mhca_stages = nn.ModuleList([
            MHCA_stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
                id_stage=idx,
            ) for idx in range(self.num_stages)
        ])

        # Classification head.
        # self.cls_head = Cls_head(embed_dims[-1], num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        if isinstance(self.init_cfg, str):
            logger = get_root_logger()
            load_checkpoint(self, self.init_cfg, map_location='cpu', strict=False, logger=logger)
            
        else:
            self.apply(self._init_weights)
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):

        # x's shape : [B, C, H, W]
        out = []
        x = self.stem(x)
        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs, idx)

            out.append(x)
            
        return out



@BACKBONES.register_module()
class afformer_base(AFFormer):
    def __init__(self, **kwargs):
        super(afformer_base, self).__init__(
                    img_size=224,
        num_stages=4,
        num_path=[1, 1, 1, 1],
        num_layers=[1, 2, 6, 2],
        embed_dims=[32, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8], **kwargs)


@BACKBONES.register_module()
class afformer_small(AFFormer):
    def __init__(self, **kwargs):
        super(afformer_small, self).__init__(
                    img_size=224,
        num_stages=4,
        num_path=[1, 1, 1, 1],
        num_layers=[1, 2, 4, 2],
        embed_dims=[32, 64, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8], **kwargs)

        

@BACKBONES.register_module()
class afformer_tiny(AFFormer):
    def __init__(self, **kwargs):
        super(afformer_tiny, self).__init__(
                    img_size=224,
        num_stages=4,
        num_path=[1, 1, 1, 1],
        num_layers=[1, 2, 4, 2],
        embed_dims=[32, 64, 160, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8], **kwargs)