import torch
from core.data.data_providers.NLSTDataProvider import DataProvider
import os
import numpy as np


class NLST3D:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.modalities = cfg.dataset.mods

        root_path = '/Learn2Reg/NLST'

        if mode == 'val' or self.cfg.dataset.one_sample_only:
            data_search_path = f'{root_path}/valid'
        elif mode == 'train':
            data_search_path = f'{root_path}/train'
        elif mode == 'test':
            data_search_path = f'{root_path}/test'

        self.dataset = DataProvider(data_search_path, 
                                    intensity_aug=cfg.dataset.intensity_aug & (self.mode == 'train'))

    def __len__(self):
        if self.mode == 'test' and self.cfg.exp.test.save_result.enable and self.cfg.exp.test.save_result.idx_sample >= 0:
            return 1
        if self.cfg.dataset.one_sample_only:
            if self.mode == 'train':
                return 500
            else:
                return 1
        else:
            return len(self.dataset)

    def __getitem__(self, item):
        if self.mode == 'test' and self.cfg.exp.test.save_result.enable and self.cfg.exp.test.save_result.idx_sample >= 0:
            item = self.cfg.exp.test.save_result.idx_sample
        if self.cfg.dataset.one_sample_only:
            return self.dataset[0]
        else:
            return self.dataset[item]

    def get_batch(self, samples):
        batch = {}

        imgs = torch.stack([torch.from_numpy(sample.pop('images')) for sample in samples]) # [B, 2, *vol_shape]
        if self.cfg.dataset.normalization == 'min-max':
            imgs = imgs - torch.amin(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
            imgs = imgs / torch.amax(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
        elif self.cfg.dataset.normalization == 'z-score':
            imgs = imgs - torch.mean(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
            imgs = imgs / torch.std(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
        else:
            raise NotImplementedError

        masks = torch.stack([torch.from_numpy(sample.pop('masks')) for sample in samples]) # [B, 2, *vol_shape]
        keypoints = [torch.from_numpy(sample.pop('keypoints')) for sample in samples] # B * [2, n, 3]

        names = [data.pop('names') for data in samples]
        batch['imgs'] = imgs
        batch['masks'] = masks
        batch['keypoints'] = keypoints
        batch['names'] = names
        batch['segs'] = masks

        return batch

    def to_device(self, batch, device):
        imgs = batch['imgs'].to(device)
        masks = batch['masks'].to(device)
        segs = batch['segs'].to(device)
        keypoints = [kpt.to(device) for kpt in batch['keypoints']]
        names = batch['names']

        # imgs: [B, 2, H, W, D]
        # segs: [B, 2, H, W, D]
        # masks: [B, 2, H, W, D]
        data = {'imgs': imgs, 'segs': segs,
                'masks': masks, 'keypoints': keypoints, 
                'names': names}
        # data = {'imgs': imgs, 'segs': segs}
        return data
