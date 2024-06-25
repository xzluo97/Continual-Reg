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
    Construct data provider for Task2 Abdomen MR-CT dataset.
    Validation dataset pattern:
    |--valid
    |  |--images
    |  |  |--AbdomenMRCT_1046_0001.nii.gz
    |  |  |--AbdomenMRCT_1047_0001.nii.gz
    |  |  |--...
    |  |--labels
    |  |  |--AbdomenMRCT_1046_0001.nii.gz
    |  |  |--AbdomenMRCT_1047_0001.nii.gz
    |  |  |--...
    |  |--masks
    |  |  |--AbdomenMRCT_1046_0001.nii.gz
    |  |  |--AbdomenMRCT_1047_0001.nii.gz
    |  |  |--...

    """
    dimension = 3
    def __init__(self, data_search_path, paired=False, **kwargs):
        self.data_search_path = data_search_path
        self.paired = paired
        self.kwargs = kwargs
        self.mr_suffix = self.kwargs.pop('mr_suffix', '0000.nii.gz')
        self.ct_suffix = self.kwargs.pop('ct_suffix', '0001.nii.gz')
        self.mr_range = self.kwargs.pop('mr_range', [-np.inf, np.inf])
        self.ct_range = self.kwargs.pop('ct_range', [-200, 300])
        self.image_prefix = self.kwargs.pop('image_prefix', 'images')
        self.label_prefix = self.kwargs.pop('label_prefix', 'labels')
        self.mask_prefix = self.kwargs.pop('mask_prefix', 'masks')
        self.label_intensities = self.kwargs.pop('label_intensities', (0, 1, 2, 3, 4))
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
        
        MR_img_names = [
            name for name in all_img_names if self.mr_suffix in os.path.basename(name)
            ]
        
        if self.paired:
            CT_img_names = [
            name.replace(self.mr_suffix, self.ct_suffix) for name in MR_img_names
            ]
            pair_names = list(zip(MR_img_names, CT_img_names))
        else:
            CT_img_names = [
            name for name in all_img_names if self.ct_suffix in os.path.basename(name)
            ]
            pair_names = list(itertools.product(MR_img_names, CT_img_names))

        return pair_names

    def __getitem__(self, item):
        pair_names = self.data_pair_names[item]
        MR_name, CT_name = pair_names

        MR_img, MR_aff, MR_head = load_image_nii(MR_name)
        CT_img, CT_aff, CT_head = load_image_nii(CT_name)

        CT_img = np.clip(CT_img, a_min=self.ct_range[0], a_max=self.ct_range[1])
        MR_img = np.clip(MR_img, a_min=None, a_max=np.percentile(MR_img, 99))
        MR_img = np.clip(MR_img, a_min=self.mr_range[0], a_max=self.mr_range[1])

        if np.min(CT_img) < 0:
            CT_img = CT_img - np.min(CT_img)
            
        if self.intensity_aug:
            CT_img = randomIntensityFilter(CT_img)
            MR_img = randomIntensityFilter(MR_img)

        CT_lab = load_image_nii(CT_name.replace(self.image_prefix, 
                                                self.label_prefix))[0]
        MR_lab = load_image_nii(MR_name.replace(self.image_prefix, 
                                                self.label_prefix))[0]

        CT_mask = load_image_nii(CT_name.replace(self.image_prefix, 
                                                 self.mask_prefix))[0]
        MR_mask = load_image_nii(MR_name.replace(self.image_prefix, 
                                                 self.mask_prefix))[0]

        images = np.stack([MR_img, CT_img]) # [2, *vol_shape]
        labels = np.stack([MR_lab, CT_lab]) # [2, *vol_shape]
        masks = np.stack([MR_mask, CT_mask]) 
        
        # crop roi
        half = np.asarray(images.shape[-self.dimension:]) // 2
        r = self.crop_shape // 2
        images = images[..., half[0] - r[0]:half[0] + r[0],
                        half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]
        labels = labels[..., half[0] - r[0]:half[0] + r[0],
                        half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]
        masks = masks[..., half[0] - r[0]:half[0] + r[0],
                      half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]

        names = [os.path.basename(name)[:-7] for name in pair_names]
        names = '_'.join(names)

        return {
            'images': images,
            'labels': labels,
            'masks': masks,
            'affines': [MR_aff, CT_aff],
            'headers': [MR_head, CT_head],
            'names': names,
        }

    def data_collate_fn(self, batch):
        AF = [data.pop('affines') for data in batch]
        HE = [data.pop('headers') for data in batch]
        batch_tensor = dict([(k, torch.stack([torch.from_numpy(data[k]) for data in batch])) for k in batch[0].keys()])
        batch_tensor['affines'] = AF
        batch_tensor['headers'] = HE

        return batch_tensor
