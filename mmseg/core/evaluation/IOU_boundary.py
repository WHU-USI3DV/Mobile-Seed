"""
This file is modified from:
https://github.com/nv-tlabs/GSCNN/blob/master/utils/f_boundary.py
and modified from:
https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code adapted from:
# https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/f_boundary.py
#
# Source License
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.s
##############################################################################
#
# Based on:
# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------
"""




import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import cv2

""" Utilities for computing, reading and saving benchmark evaluation."""

def eval_mask_boundary(seg_mask,gt_mask,num_classes,num_proc = 5,bound_th=0.005,binary = False,reduce_zero_label = False):
    """
    Compute boundary F score / IoU for a segmentation mask

    Arguments:
        seg_mask (ndarray): segmentation mask prediction
        gt_mask (ndarray): segmentation mask ground truth
        num_classes (int): number of classes

    Returns:
        F (float): mean F score across all classes
        Fpc (listof float): F score per class
    """
    p = Pool(processes=num_proc)
    # batch_size = 1
    if reduce_zero_label:
        gt_mask[gt_mask == 0] = 255
        gt_mask = gt_mask - 1
        gt_mask[gt_mask == 254] = 255
    # seg_mask = seg_mask > 0.5
    # gt_mask_total = np.zeros_like(seg_mask).astype(np.uint8)
    # gt_mask_sum = gt_mask.sum(axis = 0,keepdims = True)
    # gt_mask_total[gt_mask_sum > 255] = 255
    # gt_mask_total[np.logical_and(gt_mask_sum > 0,gt_mask_sum < 255)] = 1.0
    
    # intersection = np.zeros(num_classes,dtype = np.float32)
    # union = np.zeros(num_classes,dtype = np.float32)
    # pred = np.zeros(num_classes,dtype = np.float32)
    # gt = np.zeros(num_classes,dtype = np.float32)
    # Fpc = np.zeros(num_classes,dtype = np.float32)
    # Fc = np.zeros(num_classes,dtype = np.float32)
    # for i in range(batch_size):
    # for i in range(num_classes): 
    # for class_id in tqdm(range(num_classes)):
    if binary == False:
        args = [((seg_mask == i).astype(np.uint8), 
                 (gt_mask == i).astype(np.uint8),
                 gt_mask == 255,
                 bound_th) 
                 for i in range(num_classes)]
        temp = p.map(db_eval_boundary_wrapper,args)
        temp = np.array(temp)
        intersection = temp[:,0]
        union = temp[:,1]
        pred = temp[:,2]
        gt = temp[:,3]
        p.close()
        p.join()
        # 
    else:
        binary_gt_mask = sum(seg2bmap(gt_mask == i) for i in range(num_classes))
        binary_gt_mask = binary_gt_mask > 0
        args = [seg_mask.astype(np.uint8), 
                 binary_gt_mask.astype(np.uint8),
                 gt_mask == 255,
                 bound_th] 
        temp = db_eval_boundary_wrapper(args)
        intersection = temp[0]
        union = temp[1]
        pred = temp[2]
        gt = temp[3]
    # temp = []
    # for i in range(num_classes):
    #     temp.append(db_eval_boundary_wrapper(args[i]))
    
    # Fs = temp[:,0:1]
    # _valid = ~np.isnan(Fs)
    # Fc = np.sum(_valid,axis = 1)
    # Fs[np.isnan(Fs)] = 0
    # Fpc = np.sum(Fs,axis = -1)
    # p.close()
    # p.join()
    return [intersection,union,pred,gt]


#def db_eval_boundary_wrapper_wrapper(args):
#    seg_mask, gt_mask, class_id, batch_size, Fpc = args
#    print("class_id:" + str(class_id))
#    p = Pool(processes=10)
#    args = [((seg_mask[i] == class_id).astype(np.uint8), 
#             (gt_mask[i] == class_id).astype(np.uint8)) 
#             for i in range(batch_size)]
#    Fs = p.map(db_eval_boundary_wrapper, args)
#    Fpc[class_id] = sum(Fs)
#    return

def db_eval_boundary_wrapper(args):
    foreground_mask, gt_mask, ignore, bound_th = args
    return db_eval_boundary(foreground_mask, gt_mask,ignore, bound_th)

def db_eval_boundary(foreground_mask,gt_mask, ignore_mask = 255,bound_th= 0.00088,binary = False):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    #print(bound_pix)
    #print(gt.shape)
    #print(np.unique(gt))
    foreground_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0
    
    if binary == False:
    # Get the pixel boundaries of both masks
        fg_boundary = seg2bmap(foreground_mask)
        gt_boundary = seg2bmap(gt_mask)


    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix)) # pred boundary (dilation,binary), i
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix)) # ground-truth boundary (dilation,binary)
    # fg_dil = mask_to_boundary(foreground_mask,dilation=bound_pix)
    # gt_dil = mask_to_boundary(gt_mask,dilation=bound_pix)

    # Get the intersection
    # gt_match = gt_boundary * fg_dil # dilated grount truth & pred (for recall)
    # fg_match = fg_boundary * gt_dil # pred & dilated grount truth (for precision)
    intersection = np.sum((fg_dil * gt_dil)).astype(np.float32)
    union = np.sum((fg_dil + gt_dil)).astype(np.float32)
    pred = np.sum(fg_dil.astype(np.float32))
    gt = np.sum(gt_dil.astype(np.float32))
    
    """
    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)
    
    
    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall)
    """

    
    return [intersection,union,pred,gt]

def mask_to_boundary(mask, dilation):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    # img_diag = np.sqrt(h ** 2 + w ** 2)
    # dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=int(dilation))
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+np.floor((y-1)+height / h)
                    i = 1+np.floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap
