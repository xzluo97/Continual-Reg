import torch
from core.data.data_providers.CTCTDataProvider import DataProvider as CTCTDataProvider
from core.data.data_providers.MRCTDataProvider import DataProvider as MRCTDataProvider
from core.data.data_providers.NLSTDataProvider import DataProvider as NLSTDataProvider
from core.data.data_providers.OASISDataProvider import DataProvider as OASISDataProvider
from core.data.image_utils import get_label_center

import os
import numpy as np


# task='ctct', 'mrct', 'nlst', 'oasis'
class Continual3D:
    def __init__(self, cfg, task, mode):
        self.cfg = cfg
        self.mode = 'valid' if mode == 'val' else mode
        self.task = task

        root_path = '/Learn2Reg/'
        
        if mode == 'train':
            training = True
            oasis_max_length = 10000
        else:
            training = False
            oasis_max_length = 200

        if self.task == "ctct":
            dataset = CTCTDataProvider(f'{root_path}/AbdomenCTCT/{self.mode}', 
                                       intensity_aug=cfg.dataset.intensity_aug & (self.mode == 'train'))
        elif self.task == "mrct":
            if self.mode == 'test':
                dataset = MRCTDataProvider(f'{root_path}/AbdomenMRCT/{self.mode}', paired=True)
            else:
                dataset = MRCTDataProvider(f'{root_path}/AbdomenMRCT/{self.mode}', paired=False, 
                                           intensity_aug=cfg.dataset.intensity_aug & (self.mode == 'train'))
        elif self.task == "nlst":
            dataset = NLSTDataProvider(f'{root_path}/NLST/{self.mode}', 
                                       intensity_aug=cfg.dataset.intensity_aug & (self.mode == 'train'))
        elif self.task == "oasis":
            dataset = OASISDataProvider(f'{root_path}/OASIS/{self.mode}',
                                        training=training, max_length=oasis_max_length,
                                        intensity_aug=cfg.dataset.intensity_aug & (self.mode == 'train'))

        self.dataset = dataset

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

        imgs = torch.stack([torch.from_numpy(sample.pop('images')) for sample in samples])  # [B, 2, *vol_shape]
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
                seg = torch.from_numpy(sample.pop('labels'))
            else:
                seg = torch.from_numpy(sample.get('masks'))
            segs.append(seg)
            
            if 'keypoints' in sample:
                keypoints.append(torch.from_numpy(sample.pop('keypoints')))
            elif self.cfg.model.tre.label_center:
                keypoints.append(get_label_center(seg) * 2) # [2, n, 3]
            else:
                keypoints.append(torch.zeros(2, 1, 3))

            if 'masks' in sample:
                masks.append(torch.from_numpy(sample.pop('masks')))
            else:
                masks.append(torch.ones(2, *self.cfg.dataset.size_img))

        batch['masks'] = torch.stack(masks)
        batch['segs'] = torch.stack(segs)
        if len(keypoints) > 0:
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