# -*- coding: utf-8 -*-
"""

__author__ == Xinzhe Luo
__version__ == 0.1
"""

import glob
import itertools
import logging
import os
# import random
import numpy as np
import torch
from torch.utils.data import Dataset
from core.data.image_utils import strsort, load_image_nii
from core.data.intensity_augment import randomIntensityFilter


class DataProvider(Dataset):
    """
    Construct data provider for Task1 Abdomen CT-CT dataset.
    Validation dataset pattern:
    |--valid
    |  |--images
    |  |  |--AbdomenCTCT_0022_0000.nii.gz
    |  |  |--AbdomenCTCT_0023_0000.nii.gz
    |  |  |--...
    |  |--labels
    |  |  |--AbdomenCTCT_0022_0000.nii.gz
    |  |  |--AbdomenCTCT_0023_0000.nii.gz
    |  |  |--...

    """
    dimension = 3
    def __init__(self, data_search_path, training=False, **kwargs):
        self.data_search_path = data_search_path
        self.training = training
        self.kwargs = kwargs
        self.ct_suffix = self.kwargs.pop('ct_suffix', '0000.nii.gz')
        self.ct_range = self.kwargs.pop('ct_range', [-200, 300])
        self.image_prefix = self.kwargs.pop('image_prefix', 'images')
        self.label_prefix = self.kwargs.pop('label_prefix', 'labels')
        self.intensity_aug = self.kwargs.pop('intensity_aug', False)
        self.equalize_hist = self.kwargs.pop('equalize_hist', False)
        self.crop_shape = np.asarray(self.kwargs.pop('crop_shape', (112, 96, 112)),
                                     dtype=np.int32)

        self.data_pair_names = self._find_data_names(self.data_search_path)

    def __len__(self):
        return len(self.data_pair_names)

    def get_image_name(self, index):
        return self.data_pair_names[index]

    def _find_data_names(self, data_search_path):
        """
        Get pairs of image names.

        :param data_search_path:
        :return:
        """
        all_nii_names = strsort(glob.glob(os.path.join(data_search_path, 
                                                       '**/*.nii.gz'), 
                                          recursive=True))
        all_nii_names = [os.path.normpath(name) for name in all_nii_names]
        all_img_names = [name for name in all_nii_names if name.split(os.path.sep)[-2] == self.image_prefix]
        
        CT_img_names = [
            name for name in all_img_names if self.ct_suffix in os.path.basename(name)
            ]
        
        if self.training:
            pair_names = list(itertools.product(CT_img_names, CT_img_names))
        else:
            pair_names = list(itertools.permutations(CT_img_names, r=2))

        return pair_names

    def __getitem__(self, item):
        pair_names = self.data_pair_names[item]
        name1, name2 = pair_names

        img1, aff1, head1 = load_image_nii(name1)
        img2, aff2, head2 = load_image_nii(name2)

        img1 = np.clip(img1, a_min=self.ct_range[0], a_max=self.ct_range[1])
        img2 = np.clip(img2, a_min=self.ct_range[0], a_max=self.ct_range[1])

        if np.min(img1) < 0:
            img1 = img1 - np.min(img1)
        if np.min(img2) < 0:
            img2 = img2 - np.min(img2)
            
        if self.intensity_aug:
            img1 = randomIntensityFilter(img1)
            img2 = randomIntensityFilter(img2)

        lab1 = load_image_nii(name1.replace(self.image_prefix, self.label_prefix))[0]
        lab2 = load_image_nii(name2.replace(self.image_prefix, self.label_prefix))[0]

        images = np.stack([img1, img2]) # [2, *vol_shape]
        labels = np.stack([lab1, lab2]) # [2, *vol_shape]
        
        # crop roi
        half = np.asarray(images.shape[-self.dimension:]) // 2
        r = self.crop_shape // 2
        images = images[..., half[0] - r[0]:half[0] + r[0],
                        half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]
        labels = labels[..., half[0] - r[0]:half[0] + r[0],
                        half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]

        names = [os.path.basename(name)[:-7] for name in pair_names]
        names = '_'.join(names)

        return {
            'images': images,
            'labels': labels,
            'affines': [aff1, aff2],
            'headers': [head1, head2],
            'names': names,
        }

    def data_collate_fn(self, batch):
        AF = [data.pop('affines') for data in batch]
        HE = [data.pop('headers') for data in batch]
        batch_tensor = dict([(k, torch.stack([torch.from_numpy(data[k]) for data in batch])) for k in batch[0].keys()])
        batch_tensor['affines'] = AF
        batch_tensor['headers'] = HE

        return batch_tensor
