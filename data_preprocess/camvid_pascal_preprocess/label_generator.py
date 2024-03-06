import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import argparse

Camvid_colorlist = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]

Pascal_context59_colorlist = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
               [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
               [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
               [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
               [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
               [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
               [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
               [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
               [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
               [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
               [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
               [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
               [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
               [0, 235, 255], [0, 173, 255], [31, 0, 255]]

Pascal_context_colorlist = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                image[:, :, c] + color[c],
                                image[:, :, c])
    return image

class label_Generator():
    def __init__(self,
                 dataset,
                 root_path,
                 color_list = None,
                 ignore_label = 255,
                 reduce_zero_label = False,
                 gtori_suffix = "_L.png",
                 gtseg_suffix = "_trainIds.png",
                 gtsegedge_suffix = "_edge_bg.png",
                 colorsegedge_suffix = "_coloredge.png"):
        assert dataset == "Camvid" or dataset == "PASCAL", "for Camvid and PASCAL Context datasets only!"
        self.dataset = dataset
        self.root_path = root_path
        self.color_list = color_list
        self.ignore_label = ignore_label
        self.reduce_zero_label = reduce_zero_label
        self.gtori_suffix = gtori_suffix
        self.gtseg_suffix = gtseg_suffix
        self.gtsegedge_suffix = gtsegedge_suffix
        self.colorsegedge_suffix = colorsegedge_suffix
        
        self.num_class = len(self.color_list)
        
        self.kernel_list = [np.zeros((5,5)) for _ in range(12)]
        for i in range(12):
            self.kernel_list[i][2,2] = -1.0
        self.kernel_list[0][0,2] = 1.0
        self.kernel_list[1][1,1] = 1.0
        self.kernel_list[2][1,2] = 1.0
        self.kernel_list[3][1,3] = 1.0
        self.kernel_list[4][2,0] = 1.0
        self.kernel_list[5][2,1] = 1.0
        # self.kernel_list[5][2,2] = 0.0
        self.kernel_list[6][2,3] = 1.0
        self.kernel_list[7][2,4] = 1.0
        self.kernel_list[8][3,1] = 1.0
        self.kernel_list[9][3,2] = 1.0
        self.kernel_list[10][3,3] = 1.0
        self.kernel_list[11][4,2] = 1.0
        
        # self.image_path = os.path.join(self.root_path,"images")
        # self.gt_path = os.path.join(self.root_path,"annotations")
        if dataset == "Camvid":
            self.train_list = [line.strip().split() for line in open(os.path.join(self.split_list,"trainval.lst"))]
            self.test_list = [line.strip().split() for line in open(os.path.join(self.split_list,"test.lst"))]
            self.images_split_path = os.path.join(self.root_path,"images_tvt")
            self.gt_split_path = os.path.join(self.root_path,"annotations_tvt")
        elif dataset == "PASCAL":
            self.train_list = os.path.join(root_path,"VOC2010/ImageSets/SegmentationContext/train.txt")
            self.test_list = os.path.join(root_path,"VOC2010/ImageSets/SegmentationContext/test.txt")    
            # self.images_split_path = os.path.join(self.root_path,"images_tvt")
            self.gt_split_path = os.path.join(self.root_path,"VOC2010/SegmentationClassContext")

    def color2label(self,mode = "train"):
        # training set
        if mode == "train":
            data_list = self.train_list
        else:
            data_list = self.test_list
        for train_path in tqdm(data_list,desc = mode + "_color2label"):
            image_base_path,gt_base_path = train_path
            image_path = os.path.join(self.root_path,image_base_path)
            gt_path = os.path.join(self.root_path,gt_base_path)
            
            image_np = cv2.imread(image_path,1)
            gt_np = cv2.imread(gt_path,1)[...,::-1] # BGR --> RGB
            label_np = np.ones(gt_np.shape[:2],dtype = np.uint8) * self.ignore_label
            for i,c in enumerate(self.color_list):
                label_np[(gt_np == c).sum(-1) == 3] = i

            image_out_path = os.path.join(self.images_split_path,mode,os.path.basename(image_base_path))
            gt_out_path = os.path.join(self.gt_split_path,mode,os.path.basename(gt_base_path)).replace(self.gtori_suffix,self.gtseg_suffix)
            cv2.imwrite(image_out_path,image_np)
            cv2.imwrite(gt_out_path,label_np)
    
    def color2label_traintest(self):
        self.color2label("train")
        self.color2label("test")

    def edge2color(self, mask):
        h, w, c = mask.shape
        pred = np.unpackbits(mask,axis=2)[:,:,-1:-12:-1]
        image = np.zeros((h, w, 3))

        # image = image.astype(np.uint32)

        # pred = np.where(pred, 1, 0).astype(np.bool)
        edge_sum = np.zeros((h, w))

        for i in range(self.num_class):
            color = self.color_list[i]
            edge = pred[:,:,i]
            edge_sum = edge_sum + edge
            masked_image = apply_mask(image, edge, color)

        edge_sum = np.array([edge_sum, edge_sum, edge_sum])
        edge_sum = np.transpose(edge_sum, (1, 2, 0))
        idx = edge_sum > 0
        masked_image[idx] = masked_image[idx]/edge_sum[idx]
        masked_image[~idx] = 255

        return masked_image
        # cv2.imwrite(path,masked_image[:,:,::-1])

    def label2edge(self):
        label_list = glob(os.path.join(self.gt_split_path,'*'+self.gtseg_suffix))
        label_list.sort()
        for label_path in tqdm(label_list,desc = self.dataset):
            edge_path = label_path.replace(self.gtseg_suffix,self.gtsegedge_suffix)
            # coloredge_path = label_path.replace(self.gtseg_suffix,self.colorsegedge_suffix)
            label = cv2.imread(label_path,0)
            if self.dataset == "Camvid":
                label_edge_channel = 3
                dim = 8
                dtype = np.uint8
            else:
                # self.dataset == "PASCAL"
                label_edge_channel = 4
                dim = 16
                dtype = np.uint16
                if self.reduce_zero_label:# ignore zero label(background in PASCAL Context 59)
                    label[label == 0] = 255
                    label = label - 1
            # label_edge_channel = self.num_class // 16 + 1
            label_edge = np.zeros((label.shape[0],label.shape[1],label_edge_channel),dtype = dtype)
            for i in range(self.num_class):  
                ch = i // dim
                label_i = (label == i).astype(np.float32)
                if label_i.sum() > 0:
                    biedge = sum(abs(cv2.filter2D(label_i,ddepth=-1,kernel = kernel)) for kernel in self.kernel_list) > 0 
                    label_edge[biedge,ch] = label_edge[biedge,ch] + 2 ** (dim - 1 - i % dim)
            # color_edge = self.edge2color(label_edge)
            cv2.imwrite(edge_path,label_edge)
            # cv2.imwrite(coloredge_path,color_edge[...,::-1])
    
    def label2edge_traintest(self):
        self.label2edge("train")
        self.label2edge("test")

    def label2color(self):
        pass

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("dataset",required=True,type=str)
arg_parse.add_argument("data_path",required=True,type=str)
arg_parse.add_argument("--reduce_zero_label",action="store_true")
parser = arg_parse.parse_args()
lb = label_Generator(parser.dataset,parser.data_path,parser.reduce_zero_label)
lb.label2edge()