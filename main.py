from createdata import FugaDataset
from torch.utils.data import DataLoader
import sys
from torchvision import models
from utils import my_collate_fn
import torch
from torch.autograd import Variable
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from getmodelinstance import get_model_instance_segmentation
from config import opt
from engine import train_one_epoch
from engine import evaluate
from torchvision.models.detection import FasterRCNN

path = opt.abs_data_path
sys.path.append(path) #データパスをシステム環境に追加


#データをリードする。
train_data_set = FugaDataset(
    root = path + opt.train_path, #img file path
    anotation_root = path + opt.train_ano, #anotation path
    train=True
                )

test_data_set = FugaDataset(
    root = path + opt.train_path, #img file path
    anotation_root = path + opt.train_ano, #anotation path
    train=False
                )


#データをiter化にする
train_dataloader = DataLoader(train_data_set, 
                                shuffle=True, 
                                batch_size=1, 
                                drop_last=True, 
                                num_workers=0, 
                                collate_fn=my_collate_fn)

test_dataloader = DataLoader(test_data_set, 
                                shuffle=False, 
                                batch_size=1, 
                                drop_last=True, 
                                num_workers=0, 
                                collate_fn=my_collate_fn)




"""
for step, (img, targets) in enumerate(train_dataloader):
    if step <= 2:
        
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
        img = img   
        print(targets) 
        break 
"""

num_classes = opt.number_class 
model = get_model_instance_segmentation(opt.number_class)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]

#最適手法を定義
optimizer = torch.optim.SGD(
    params, 
    lr=opt.lr,
    momentum=0.9, #alpha係数 
    weight_decay=0.0005 
    )

#段階的にlr調整
lr_scheduler = \
    torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
        )


num_epochs = 1
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    evaluate(model, test_dataloader, device=device)

