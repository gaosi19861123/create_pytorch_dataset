from createdata import FugaDataset
from torch.utils.data import DataLoader
import sys
from torchvision import models
from utils import my_collate_fn
import torch
from torch.autograd import Variable
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

path = "C:/Users/dso-s.gao/Desktop/signate"
sys.path.append(path)


#データをリードする。
data_set = FugaDataset(
    root = path + "/dtc_train/", #img file path
    anotation_root = path + "/dtc_train_annotations/dtc_train_annotations/", #anotation path
                )

train_dataloader = DataLoader(data_set, 
                                shuffle=True, 
                                batch_size=1, 
                                drop_last=True, 
                                num_workers=0, 
                                collate_fn=my_collate_fn)

for step, (img, target) in enumerate(train_dataloader):
    if step <= 2:
        print(target)   
        break 

detection_model = models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, 
    progress=True,
    )

num_classes = 10
in_features=  detection_model.roi_heads.box_predictor.cls_score.in_features
detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

