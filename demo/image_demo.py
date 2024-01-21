# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from glob import glob
import os
from tqdm import tqdm

def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] + color[c],
                                  image[:, :, c])
    return image

def visualize_prediction(path, pred):
    n, h, w = pred.shape
    image = np.zeros((h, w, 3))
    # image = image.astype(np.uint32)

    colors = [[128, 64, 128],
               [244, 35, 232],
               [70, 70, 70],
               [102, 102, 156],
               [190, 153, 153],
               [153, 153, 153],
               [250, 170, 30],
               [220, 220, 0],
               [107, 142, 35],
               [152, 251, 152],
               [70, 130, 180],
               [220, 20, 60],
               [255, 0, 0],
               [0, 0, 142],
               [0, 0, 70],
               [0, 60, 100],
               [0, 80, 100],
               [0, 0, 230],
               [119, 11, 32]]

    # pred = np.where(pred >= 0.5, 1, 0)
    boundary_sum = np.zeros((h, w))

    for i in range(n):
      color = colors[i]
      boundary = pred[i,:,:]
      boundary_sum = boundary_sum + boundary
      masked_image = apply_mask(image, boundary, color)

    boundary_sum = np.array([boundary_sum, boundary_sum, boundary_sum])
    boundary_sum = np.transpose(boundary_sum, (1, 2, 0))
    idx = boundary_sum > 0
    masked_image[idx] = masked_image[idx]/boundary_sum[idx]
    masked_image[~idx] = 255
    
    cv2.imwrite(path,masked_image[...,::-1])

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask

def onehot_to_multiclass_boundarys(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an multi-class boundary map (K,H,W)

    """
    
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='reflect')
    
    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)
        
    return np.array(channels)

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('out_seg',help='Path to output segment file')
    parser.add_argument('--out_sebound',help='Path to output semantic boundary file')
    parser.add_argument('--out_bibound',help='Path to output binary boundary file')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()


    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    result = inference_segmentor(model, args.img)
    seg_pred,bound_pred = result
        # show the results
        # reset outfile name
        # args.out_folder = args.img.replace("color","seq_pred")
        # args.outss_file = args.out_file.replace(".png",outss_suffix)
        # args.outse_file = args.outss_file.replace(outss_suffix,outse_suffix)
        # args.outed_file = args.outss_file.replace(outss_suffix,outed_suffix)
        
        # cv2.imwrite(outcolor_path,cv2.imread(i))
    show_result_pyplot(
        model,
        args.img,
        seg_pred,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.out_seg)
    
    if args.out_sebound:
        onehot_mask = mask_to_onehot(seg_pred[0],19) # one input img in default
        sebound_mask = onehot_to_multiclass_boundarys(onehot_mask,2,19)
        visualize_prediction(args.out_sebound,sebound_mask)
    
    if args.out_bibound:
        bound_pred = (bound_pred[0] * 255.0).astype(np.uint8)
        bound_pred = cv2.applyColorMap(bound_pred,13)
        cv2.imwrite(args.out_bibound,bound_pred)



if __name__ == '__main__':
    main()
