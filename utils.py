#object検知時に自作の成形関数を導入
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

