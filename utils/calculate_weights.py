import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    classes_weights_path1 = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights1.npy')
    np.save(classes_weights_path, ret)
    np.save(classes_weights_path1, z)

    return ret


def calculate_weigths_labels_for_all(train_loader, test_loader, num_classes):
    # Create an instance from the data loader
    z = None
    # Initialize tqdm
    tqdm_batch = tqdm(train_loader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        # mask = (y >= 0) & (y < num_classes)
        # labels = y[mask].astype(np.uint8)
        count_l = np.bincount(y.astype(np.uint8).flatten(), minlength=num_classes)
        if z is None:
            z = count_l
        else:
            z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    print("Total Training Pixels: ", total_frequency)
    print("Frequency of Training Pixels: ", z)


    # Create an instance from the data loader
    z = None
    # Initialize tqdm
    tqdm_batch = tqdm(test_loader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        # mask = (y >= 0) & (y < num_classes)
        # labels = y[mask].astype(np.uint8)
        count_l = np.bincount(y.astype(np.uint8).flatten(), minlength=num_classes)
        if z is None:
            z = count_l
        else:
            z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    print("Total Testing Pixels: ", total_frequency)
    print("Frequency of Testing Pixels: ", z)

    # return ret