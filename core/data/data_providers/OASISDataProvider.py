# -*- coding: utf-8 -*-
"""

__author__ == Xinzhe Luo
__version__ == 0.1
"""

import glob
import itertools
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from core.data.image_utils import strsort, load_image_nii
from core.data.intensity_augment import randomIntensityFilter


class DataProvider(Dataset):
    """
    Construct data provider for Task4 OASIS dataset.
    Validation dataset pattern:
    |--valid
    |  |--images
    |  |  |--OASIS_0291_0000.nii.gz
    |  |  |--OASIS_0292_0000.nii.gz
    |  |  |--...
    |  |--labels
    |  |  |--OASIS_0291_0000.nii.gz
    |  |  |--OASIS_0292_0000.nii.gz
    |  |  |--...
    |  |--masks
    |  |  |--OASIS_0291_0000.nii.gz
    |  |  |--OASIS_0292_0000.nii.gz
    |  |  |--...

    """
    dimension = 3
    def __init__(self, data_search_path, training=True, **kwargs):
        self.data_search_path = data_search_path
        self.training = training
        self.kwargs = kwargs
        self.mr_suffix = self.kwargs.pop('mr_suffix', '0000.nii.gz')
        self.mr_range = self.kwargs.pop('mr_range', [-np.inf, np.inf])
        self.image_prefix = self.kwargs.pop('image_prefix', 'images')
        self.label_prefix = self.kwargs.pop('label_prefix', 'labels')
        self.mask_prefix = self.kwargs.pop('mask_prefix', 'masks')
        self.intensity_aug = self.kwargs.pop('intensity_aug', False)
        self.equalize_hist = self.kwargs.pop('equalize_hist', False)
        self.pad_shape = np.asarray(self.kwargs.pop('crop_shape', (112, 96, 112)),
                                    dtype=np.int32)
        self.max_length = self.kwargs.pop('max_length', 10000)

        self.data_pair_names = self._find_data_names(self.data_search_path)

    def __len__(self):
        return min(len(self.data_pair_names), self.max_length)

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
        
        MR_img_names = [
            name for name in all_img_names if self.mr_suffix in os.path.basename(name)
            ]
        
        if self.training:
            pair_names = list(itertools.product(MR_img_names, MR_img_names))
        else:
            pair_names = list(itertools.permutations(MR_img_names, 2))

        if len(pair_names) > self.max_length:
            random.seed(42)
            pair_names = random.sample(pair_names, self.max_length)

        return pair_names

    def __getitem__(self, item):
        pair_names = self.data_pair_names[item]
        name1, name2 = pair_names

        img1, aff1, head1 = load_image_nii(name1)
        img2, aff2, head2 = load_image_nii(name2)

        img1 = np.clip(img1, a_min=None, a_max=np.percentile(img1, 99))
        img1 = np.clip(img1, a_min=self.mr_range[0], a_max=self.mr_range[1])
        img2 = np.clip(img2, a_min=None, a_max=np.percentile(img2, 99))
        img2 = np.clip(img2, a_min=self.mr_range[0], a_max=self.mr_range[1])
        if self.intensity_aug:
            img1 = randomIntensityFilter(img1)
            img2 = randomIntensityFilter(img2)

        lab1 = load_image_nii(name1.replace(self.image_prefix, self.label_prefix))[0]
        lab2 = load_image_nii(name2.replace(self.image_prefix, self.label_prefix))[0]

        mask1 = load_image_nii(name1.replace(self.image_prefix, self.mask_prefix))[0]
        mask2 = load_image_nii(name2.replace(self.image_prefix, self.mask_prefix))[0]

        images = np.stack([img1, img2]).transpose((0, 1, 3, 2)) # [2, *vol_shape]
        labels = np.stack([lab1, lab2]).transpose((0, 1, 3, 2)) # [2, *vol_shape]
        masks = np.stack([mask1, mask2]).transpose((0, 1, 3, 2)) 
                
        ori_shape = np.asarray(images.shape[-self.dimension:])
        widths = (self.pad_shape - ori_shape) // 2
        images = np.pad(images, 
                        pad_width=((0, 0), 
                                   (widths[0], self.pad_shape[0] - widths[0] - ori_shape[0]), 
                                   (widths[1], self.pad_shape[1] - widths[1] - ori_shape[1]),
                                   (widths[2], self.pad_shape[2] - widths[2] - ori_shape[2]))
                        )
        labels = np.pad(labels, 
                        pad_width=((0, 0), 
                                   (widths[0], self.pad_shape[0] - widths[0] - ori_shape[0]), 
                                   (widths[1], self.pad_shape[1] - widths[1] - ori_shape[1]),
                                   (widths[2], self.pad_shape[2] - widths[2] - ori_shape[2]))
                        )
        masks = np.pad(masks, 
                       pad_width=((0, 0), 
                                  (widths[0], self.pad_shape[0] - widths[0] - ori_shape[0]), 
                                  (widths[1], self.pad_shape[1] - widths[1] - ori_shape[1]),
                                  (widths[2], self.pad_shape[2] - widths[2] - ori_shape[2]))
                        )
                        
        names = [os.path.basename(name)[:-7] for name in pair_names]
        names = '_'.join(names)

        return {
            'images': images,
            'labels': labels,
            'masks': masks,
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
