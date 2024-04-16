import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch.utils.data as data_utils

import numpy as np

import os


class ExpertID_Dataset(Dataset):
    """Custom Dataset for loading NumPy array data"""

    def __init__(self, x5: np.ndarray, x7: np.ndarray, target: np.ndarray, mode, data_stat, test_normalise=True):
        """
        Args:
            data_array (numpy array): A numpy array of shape (num_data_points, 1, 12, 12).
            target: numpy array of shape (num_data_points,)
        """
        assert len(x5.shape) == 4, f"wrong data shape: {x5.shape}"

        sorted_target = np.sort(target)
        target_map = dict()
        i = 0
        for c in sorted_target:
            if c not in target_map.keys():
                target_map[c] = i
                i += 1

        data = np.zeros((x5.shape[0], 24, 24, 1))
        data[:, 0:12, 0:12, 0] = x5[:, :, :, 0]
        data[:, 0:12, 12:, 0] = x5[:, :, :, 1]
        data[:, 12:, 0:12, 0] = x7[:, :, :, 0]
        data[:, 12:, 12:, 0] = x7[:, :, :, 1]
        # print(data.shape)
        data = np.transpose(data, (0, 3, 1, 2))
        # Convert the data to torch.FloatTensor as it is the most common dtype for images
        self.data_array = torch.tensor(data, dtype=torch.float32)
        
        for i, _ in enumerate(target):  # transform target such that the class labels start from 0 to n_classes-1 inclusive
            target[i] = target_map[target[i]]
        self.target = torch.tensor(np.array(target, dtype=np.float32), dtype=torch.int64)
        if mode == "train":
            self.transform =  transforms.Compose([
                # transforms.ToTensor(),  ## data augmentation for training only
                transforms.RandomCrop(size=(24, 24), padding=4, pad_if_needed=False, padding_mode='edge'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(*data_stat)
                ])
        elif mode == "test":
            if test_normalise == False:
                self.transform =  transforms.Compose([
                    # transforms.ToTensor(),  ## data augmentation below for training only
                    # transforms.Normalize(*data_stat)
                    ])
            else:
                self.transform =  transforms.Compose([
                    # transforms.ToTensor(),  ## data augmentation below for training only
                    transforms.Normalize(*data_stat)
                    ])
        else:
            raise Exception("wrong mode, choose from 'train' and 'test'")

    def __len__(self):
        """Return the total number of data points in the dataset."""
        return self.data_array.shape[0]

    def __getitem__(self, idx):
        image = self.transform(self.data_array[idx])
        target = self.target[idx]
        return image, target

"""std num classes = 4 2.267659056475096
4.469289374325782
std num classes = 64 2.296275276853656
4.5146239704420505
std num classes = 2 2.2193135572435514
4.587178191246345
std num classes = 16 2.3092031327372786
4.460865931701998
std num classes = 8 2.2793343532940935
4.421694285158908
std num classes = 32 2.2737503717264227
4.567340335655347
std num classes = 128 2.3167054615253844
4.517264099638317
std num classes = 512 2.299420980057555
4.499429311338581
std num classes = 256 2.2901221928285342
4.485078938202259
std num classes = 1000 2.303071876769264
4.495041399588444
"""


# input shape of data: (num_data_points, 12, 12, 2)
# desired shape of data: torch.rand((num_data_points, 1, 24, 24))
# this is computed using all training data. did not include any test data
num_class_to_stats = {4: ((4.469289374325782,), (2.267659056475096,)), 
                      64: ((4.5146239704420505,), (2.296275276853656,)),
                      2: ((4.587178191246345,), (2.2193135572435514,)), 
                      16: ((4.460865931701998,), (2.3092031327372786,)), 
                      8: ((4.421694285158908,), (2.2793343532940935,)), 
                      32: ((4.567340335655347,), (2.2737503717264227,)), 
                      128: ((4.517264099638317,), (2.3167054615253844,)), 
                      256: ((4.485078938202259,), (2.2901221928285342,)), 
                      512: ((4.499429311338581,), (2.299420980057555,)), 
                      1000: ((4.495041399588444,), (2.303071876769264,))}


def get_data_loader(mode, batch_size, num_classes, shuffle=True, test_normalise=False):
    class_idx_dir="/home/zl310/cs585_project/vmoe/chosen_class_idx/"
    which_classes = set(np.load(os.path.join(class_idx_dir, f"n_{num_classes}.npy")))
    data_dir = f"/home/zl310/cs585_project/vmoe/{mode}_data_selected_classes/"  # unencrypted side-channel
    this_dest_dir = os.path.join(data_dir, f"data_n_{len(which_classes)}")
    x5 = np.load(os.path.join(this_dest_dir, f"x5_n_{len(which_classes)}.npy"))
    x7 = np.load(os.path.join(this_dest_dir, f"x7_n_{len(which_classes)}.npy"))
    target = np.load(os.path.join(this_dest_dir, f"y_n_{len(which_classes)}.npy"))
    data_set = ExpertID_Dataset(x5, x7, target, mode=mode, data_stat=num_class_to_stats[num_classes], test_normalise=test_normalise)
    if mode == "train":
        train_set, val_set = data_utils.random_split(data_set, [0.9, 0.1])
        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=4), \
                DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    elif mode == "test":
        return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)