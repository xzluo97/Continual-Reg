import torch
from core.data.data_providers.CTCTDataProvider import DataProvider as CTCTDataProvider
from core.data.data_providers.MRCTDataProvider import DataProvider as MRCTDataProvider
from core.data.data_providers.NLSTDataProvider import DataProvider as NLSTDataProvider
from core.data.data_providers.OASISDataProvider import DataProvider as OASISDataProvider
from torch.utils.data import ConcatDataset

import os
import numpy as np


class MultiTask3D:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode

        root_path = '/datasets/Learn2Reg/'

        if mode == 'val' or self.cfg.dataset.one_sample_only:
            dataset1 = CTCTDataProvider(f'{root_path}/AbdomenCTCT/valid')
            dataset2 = MRCTDataProvider(f'{root_path}/AbdomenMRCT/valid', paired=False)
            dataset3 = NLSTDataProvider(f'{root_path}/NLST/valid')
            dataset4 = OASISDataProvider(f'{root_path}/OASIS/valid', training=False)
        elif mode == 'train':
            dataset1 = CTCTDataProvider(f'{root_path}/AbdomenCTCT/train', 
                                        intensity_aug=cfg.dataset.intensity_aug)
            dataset2 = MRCTDataProvider(f'{root_path}/AbdomenMRCT/train', paired=False,
                                        intensity_aug=cfg.dataset.intensity_aug)
            dataset3 = NLSTDataProvider(f'{root_path}/NLST/train',
                                        intensity_aug=cfg.dataset.intensity_aug)
            dataset4 = OASISDataProvider(f'{root_path}/OASIS/train',
                                         intensity_aug=cfg.dataset.intensity_aug)
        elif mode == 'test':
            dataset1 = CTCTDataProvider(f'{root_path}/AbdomenCTCT/test')
            dataset2 = MRCTDataProvider(f'{root_path}/AbdomenMRCT/test', paired=True)
            dataset3 = NLSTDataProvider(f'{root_path}/NLST/test')
            dataset4 = OASISDataProvider(f'{root_path}/OASIS/test', training=False)

        self.datasets = [dataset1, dataset2, dataset3, dataset4]
        self.cum_length = np.cumsum([0] + [len(dataset) for dataset in self.datasets])
        self.dataset = ConcatDataset(self.datasets)

    def __len__(self):
        if self.mode == 'test' and self.cfg.exp.test.save_result.enable and self.cfg.exp.test.save_result.idx_sample >= 0:
            return 1
        if self.cfg.dataset.one_sample_only:
            if self.mode == 'train':
                return 500
            else:
                return 1
        else:
            return self.cum_length[-1]

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
        
        batch['imgs'] = imgs
        
        masks = []
        segs = []
        keypoints = []
        for sample in samples:
            if 'labels' in sample:
                segs.append(torch.from_numpy(sample.pop('labels')))
            else:
                segs.append(torch.from_numpy(sample.get('masks')))
                
            if 'keypoints' in sample:
                keypoints.append(torch.from_numpy(sample.pop('keypoints')))
            else:
                keypoints.append(torch.zeros(2, 1, 3))
                
            if 'masks' in sample:
                masks.append(torch.from_numpy(sample.pop('masks')))
            else:
                masks.append(torch.ones(2, *self.cfg.dataset.size_img))
                
        batch['masks'] = torch.stack(masks)
        batch['segs'] = torch.stack(segs)
        batch['keypoints'] = keypoints

        names = [data.pop('names') for data in samples]
        batch['names'] = names

        return batch

    def to_device(self, batch, device):
        imgs = batch['imgs'].to(device)
        names = batch['names']

        data = {'imgs': imgs, 'names': names}
        if 'masks' in batch:
            masks = batch['masks'].to(device)
            data.update({'masks': masks})
        if 'segs' in batch:
            segs = batch['segs'].to(device)
            data.update({'segs': segs})
        if 'keypoints' in batch:
            keypoints = [kpt.to(device) for kpt in batch['keypoints']]
            data.update({'keypoints': keypoints})

        # imgs: [B, 2, H, W, D]
        # segs: [B, 2, H, W, D]
        # masks: [B, 2, H, W, D]
        # keypoints: [B, 2, n, 3]
        return data
