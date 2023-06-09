from dataloaders.datasets import multimodal_dataset    #, cityscapes, coco, combine_dbs, pascal, sbd, multimodal_dataset
from torch.utils.data import DataLoader
from dataloaders.datasets import mcubes_dataset
# from dataloaders.datasets import nyudv2


def make_data_loader(args, **kwargs):

    # if args.dataset == 'pascal':
    #     train_set = pascal.VOCSegmentation(args, split='train')
    #     val_set = pascal.VOCSegmentation(args, split='val')
    #     if args.use_sbd:
    #         sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #         train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None

    #     return train_loader, val_loader, test_loader, num_class

    # elif args.dataset == 'cityscapes':
    #     train_set = cityscapes.CityscapesSegmentation(args, split='train')
    #     val_set = cityscapes.CityscapesSegmentation(args, split='val')
    #     test_set = cityscapes.CityscapesSegmentation(args, split='test')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    #     return train_loader, val_loader, test_loader, num_class

    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class

    print(args.dataset)
    if args.dataset == 'multimodal_dataset':
        train_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='train')
        val_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='val')
        test_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='test')
        # test_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='visualize')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


def make_data_loader2(args, **kwargs):
    print(args.dataset)
    if args.dataset == 'multimodal_dataset':
        train_set = mcubes_dataset.MCubeSDataset(args, split='train', nir_dim=3)
        val_set = mcubes_dataset.MCubeSDataset(args, split='val', nir_dim=3)
        test_set = mcubes_dataset.MCubeSDataset(args, split='test', nir_dim=3)
        # test_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='visualize')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    
    # elif args.dataset == 'nyudv2':
    #     train_set = nyudv2.NYUDv2(args, split='train')
    #     test_set = nyudv2.NYUDv2(args, split='test')

    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    #     return train_loader, test_loader, num_class

    else:
        raise NotImplementedError