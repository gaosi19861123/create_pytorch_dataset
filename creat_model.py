from createdata import FugaDataset
from torch.utils.data import DataLoader
import sys

path = "C:/Users/dso-s.gao/Desktop/signate"
sys.path.append(path)

data_set = FugaDataset(
    root = path + "/dtc_train_images_0/dtc_train_images/", 
    anotation_root = path + "/dtc_train_annotations/dtc_train_annotations/",
                )

#object検知時に自作のデータセット関数を導入
def my_collate_fn(batch):
    # datasetの出力が
    # [image, target] = dataset[batch_idx]
    # の場合.
    images = []
    boxes = []
    category = [] 
    for sample in batch:
        image, box, cate = sample
        images.append(image)
        boxes.append(box)
        category.append(cate)
        
    images = torch.stack(images, dim=0)
    return [images, boxes, category]

train_dataloader = DataLoader(data_set, shuffle=True, 
                                batch_size=2, 
                                drop_last=True, 
                                num_workers=0, 
                                collate_fn=my_collate_fn)