#object検知時に自作の成形関数を導入
import torch 
import json

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


def read_json_file(file_name):
    with open(file_name, "rb") as f:
        file = json.load(f)

    address = file["attributes"]['route']
    time = file["attributes"]['timeofday']

    bbox = [row["box2d"] for row in file["labels"]]
    category = [row["category"] for row in file["labels"]]
    
    return address, time, bbox, categoryS