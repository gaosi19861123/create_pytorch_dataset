import torch
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch as t

import cv2
import PIL
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import json
from collections import Counter
from categorydict import dict_category
from utils import read_json_file

class FugaDataset(object):
    
    
    def __init__(self, root, anotation_root, transforms=True, 
                train=True, category_dict=dict_category):    
        self.root = root
        self.anotation_root = anotation_root
        self.category_dict = category_dict
        self.transforms = transforms
        # 下载所有图像文件，为其排序
        # 确保它们对齐
        self.imgs = list(sorted(os.listdir(root)))
        self.anotation = list(sorted(os.listdir(anotation_root)))
    
    def __getitem__(self, index):
        #画像データ、PILで読み、TO_TENSORで返す
        self.img_path = os.path.join( self.root + self.imgs[index] )
        img = Image.open(self.img_path).convert("RGB")
        json = self.anotation[index]
        
        #画像を編集するときであれば
        #The input to the model is expected to be a list of tensors, 
        #each of shape [C, H, W], one for each image, and should be in 0-1 range. 
        #Different images can have different sizes.
        
        if self.transforms:
            transform = transforms.Compose([
                            #transforms.CenterCrop((100, 100)), #中心クロップ
                            #transforms.Grayscale(num_output_channels=1), #灰色化
                            #transforms.RandomHorizontalFlip(), #水平反転
                            #transforms.Scale(224), #resize
                            #transforms.TenCrop(3), # 十分割
                            #transforms.Lambda(lambda crops : t.stack([transforms.ToTensor()(crop) for crop in crops])),
                            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], #標準化
                            #std = [ 0.229, 0.224, 0.225 ]),
                            #transforms.ToPILImage(mode="RGBA") 
                            transforms.ToTensor() #テンソル化
                            ])
            img = transform(img)
        
        #データのBBOXを書き込み
        address, time, bbox, category = read_json_file(self.anotation_root + self.anotation[index])
        
        #boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
        #with values between 0 and H and 0 and W
        #boxes の座標をembedingします。
        num_objs = len(bbox)
        boxes = []
        for i in range(num_objs):
            x1 = bbox[i]["x1"]
            x2 = bbox[i]["x2"]
            y1 = bbox[i]["y1"]
            y2 = bbox[i]["y2"]
            boxes.append([x1, y1, x2, y2])
        
        #bounding_boxをテンソル化
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        #categoryをテンソル化します。
        category_transed = [self.category_dict[cate] for cate in category]
        category_transed = torch.as_tensor(category_transed, dtype=torch.int)
 
        return img, boxes, category_transed
    
    def __len__(self):
        return len(self.imgs)

