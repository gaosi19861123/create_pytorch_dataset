def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
    # our dataset has two classes only - background and person
    num_classes = opt.number_class
    # use our dataset and defined transformations
    data_set = FugaDataset(
        root = path + opt.train_path, #img file path
        anotation_root = path + opt.train_ano, #anotation path
                )
    
    #split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    # define training and validation data loaders
    data_loader = DataLoader(data_set, 
                            shuffle=True, 
                            batch_size=1, 
                            drop_last=True, 
                            num_workers=0, 
                            collate_fn=my_collate_fn)
 
    #data_loader_test = torch.utils.data.DataLoader(
    #    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    #    collate_fn=utils.collate_fn)
 
    # get the model using our helper function
    model = get_model_instance_segmentation(opt.number_class)
 
    # move model to the right device
    model.to(device)
 
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=opt.lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    # let's train it for 10 epochs
    #num_epochs = 10
 
    #for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
    #    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
    #    lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)
 
    print("That's it!")

    main()