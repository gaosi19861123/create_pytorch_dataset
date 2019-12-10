from createdata import FugaDataset
from torch.utils.data import DataLoader
import sys
from utils import my_collate_fn

path = "C:/Users/dso-s.gao/Desktop/signate"
sys.path.append(path)


#データをリードする。
data_set = FugaDataset(
    root = path + "/dtc_train_images_0/dtc_train_images/", 
    anotation_root = path + "/dtc_train_annotations/dtc_train_annotations/",
                )

train_dataloader = DataLoader(data_set, shuffle=True, 
                                batch_size=2, 
                                drop_last=True, 
                                num_workers=0, 
                                collate_fn=my_collate_fn)
