import os
import numpy as np
import rasterio
from torch.utils.data import Dataset
import random


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif.read().astype(np.float32)


""" Dw_S1 data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the Dw_S1 data set
    split:              str, in [all | train | test]
    sample_type:        str, [generic | cloudy_cloudfree]
    n_input_samples:    int, number of input samples in time series

    OUT:
    data_loader:        Dw_S1 instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""


class DW_S1(Dataset):
    def __init__(self, root, split="all", sample_type='nearest_time', n_input_samples=3):

        self.root_dir = root  # set root directory which contains all ROI

        assert split in ['all', 'train', 'test'], \
            "Input dataset must be either assigned as all, train or test"
        assert sample_type in ['generic', 'cloudy_cloudfree', 'nearest_time'], \
            "Input data must be either generic or cloudy_cloudfree or nearest_time type!"
        self.split = split
        self.sample_type = sample_type
        # specifies the number of samples, if only part of the time series is used as an input
        self.n_input_t = n_input_samples
        self.paths = self.get_paths()
        self.n_samples = len(self.paths)

    # indexes all patches contained in the current data split
    def get_paths(self):
        all_list = os.listdir(os.path.join(self.root_dir, 'DW', 'target'))
        random.seed(0)
        random.shuffle(all_list)
        train_num = int(len(all_list) * 0.7)
        train_list = all_list[:train_num]
        test_list = all_list[train_num:]

        if self.split == 'all':
            return all_list
        elif self.split == 'train':
            return train_list
        else:
            return test_list

    def __getitem__(self, pdx):  # get the time series of one patch

        # this returns n_input_t DW & S1 observations in the same period of every time points
        if self.sample_type == 'nearest_time':
            target_dw_path = os.path.join(self.root_dir, 'DW', 'target', self.paths[pdx])
            target_s1_path = os.path.join(self.root_dir, 'S1', 'target', self.paths[pdx])
            target_s1, target_dw = read_tif(target_s1_path), read_tif(target_dw_path)

            input_s1, input_dw, input_masks = [], [], []
            for i in range(self.n_input_t):
                input_dw_path = os.path.join(self.root_dir, 'DW', 'input{}'.format(i + 1), self.paths[pdx])
                input_s1_path = os.path.join(self.root_dir, 'S1', 'input{}'.format(i + 1), self.paths[pdx])
                input_mask_path = os.path.join(self.root_dir, 'Mask', 'input{}'.format(i + 1), self.paths[pdx])
                input_dw.append(read_tif(input_dw_path))
                input_s1.append(read_tif(input_s1_path))
                input_masks.append(read_tif(input_mask_path).astype(np.uint8))

            sample = {'input': {'DW': list(input_dw),
                                'S1': list(input_s1),
                                'masks': list(input_masks),
                                'coverage': [np.mean(mask) for mask in input_masks]
                                },
                      'target': {'DW': [target_dw],
                                 'S1': [target_s1],
                                 },
                      'patch_id': int(os.path.basename(self.paths[pdx]).split('.')[0])
                      }

        else:
            raise Exception('sample_type should be cloudy_cloudfree or generic or nearest_time')

        return sample

    def __len__(self):
        # length of generated list
        return self.n_samples
