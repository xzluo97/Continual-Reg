"""
__author__ = Xinzhe Luo
__version__ = 0.1

"""

import os
import itertools
import pandas as pd
import monai.metrics
from collections import OrderedDict
from omegaconf import OmegaConf as Ocfg
import torch
# from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from core.register.SpatialTransformer import SpatialTransformer, ResizeTransform, sub2ind
from core.register.LocalDisplacementEnergy import JacobianDeterminant
from core.metrics.Overlap import OverlapMetrics
from core.data.image_utils import save_image_nii, get_img_grid


class Base(nn.Module):
    def __init__(self, cfg: Ocfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.var.obj_model = self
        self.paths_file_net = ['core/models/base.py']
        
        self.dice = OverlapMetrics(type='all_foreground_dice')
        self.jacobian = JacobianDeterminant(dimension=cfg.dataset.dim)
        
        # [B, 2, ...], distorted images
        self.imgs_distorted: torch.Tensor = None

        # [B, 2, ...], distorted foreground masks
        self.masks_distorted: torch.Tensor = None
        
        # [B, 2, ...]
        self.segs_distorted: torch.Tensor = None
        self.segs_warped: torch.Tensor = None
        
        # [B, 2, n, dim]
        self.keypoints_distorted: torch.Tensor = None
        self.keypoints_warped: torch.Tensor = None
        
    def forward(self, data):
        self.data = data
        self.cfg.var.n_samples = b = data['imgs'].shape[0]
        if 'names' in data:
            names = data['names']
            self.cfg.var.dataset_names = [name.split('_')[0] for name in names]
        
        imgs = data['imgs']  # [B, 2, ...]
        self.cfg.var.img_size = imgs.shape[-self.cfg.dataset.dim:]
        self.imgs_distorted = imgs
        self.fixed_img = imgs[:, [0]]
        self.moving_img = imgs[:, [1]]
        
        warper = SpatialTransformer(size=imgs.shape[-self.cfg.dataset.dim:])
        self.warper = warper.to(imgs.device)
        
        if 'masks' in data:
            masks = data['masks']  # [B, 2, ...]
        else:
            masks = torch.ones_like(imgs)
        self.masks_distorted = masks
        self.fixed_mask = masks[:, [0]]
        self.moving_mask = masks[:, [1]]
        
        self.disp = self.net(self.imgs_distorted * self.masks_distorted)
        
        self.warped_moving_img = self.net.decoder.warped_moving_img
        self.imgs_warped = torch.cat([self.fixed_img, 
                                      self.warped_moving_img], dim=1)
        
        if 'segs' in data:
            segs = data['segs']  # [B, 2, ...]
            self.cfg.var.num_classes = torch.amax(segs).long().item() + 1
            self.segs_distorted = segs
            self.fixed_seg = segs[:, [0]]
            self.moving_seg = segs[:, [1]]

            self.warped_moving_seg = self.warper(self.moving_seg, self.disp, 
                                                 interp_mode='nearest', 
                                                 padding_mode='border')
            self.segs_warped = torch.cat([self.fixed_seg, self.warped_moving_seg], dim=1)
        
        if 'keypoints' in data:
            keypoints = data['keypoints'] # B * [2, n, 3]
            self.keypoints_distorted = keypoints
            self.fixed_kpt = [kpt[[0]] for kpt in keypoints] # B * [1, n, 3]
            self.moving_kpt = [kpt[[1]] for kpt in keypoints]
            
            self.resize = ResizeTransform(dimension=self.cfg.dataset.dim, 
                                          factor=2)
            disp_resize = self.resize(self.disp) # [B, 3, nx, ny, nz]
            
            self.warped_fixed_kpt = []
            for i in range(b):
                fixed_kpt_disp = torch.gather(
                    input=disp_resize[i].view(self.cfg.dataset.dim, -1), 
                    index=sub2ind(vol_shape=disp_resize.shape[-self.cfg.dataset.dim:], subs=self.fixed_kpt[i]).repeat(self.cfg.dataset.dim, 1),
                    dim=1)  # [3, n]
                self.warped_fixed_kpt.append(self.fixed_kpt[i] + fixed_kpt_disp.permute(1, 0).unsqueeze(0))
            
            self.keypoints_warped = [torch.cat([self.warped_fixed_kpt[i], 
                                                self.moving_kpt[i]], dim=0) for i in range(b)]
        
        return self.disp
    
    def before_epoch(self, mode='train', i_repeat=0):
        self.mode = mode
        if mode == 'train':
            self.metrics_epoch_train = OrderedDict()
        else:
            self.metrics_epoch_val = OrderedDict()
        self.idx_sample = 0

        if mode == 'test':
            self.metrics_samplewise = {}
            self.metrics_samplewise['dice'] = []
            self.metrics_samplewise['assd'] = []
            self.metrics_samplewise['tre'] = []
            self.metrics_samplewise['names'] = []
        
    def after_epoch(self, mode='train'):
        if 'task_sequential' in self.cfg.model and self.cfg.model.task_sequential:
            dataset = getattr(self.cfg.var.obj_operator, f'{mode}_sets')[self.cfg.var.obj_operator.task_idx]
        else:
            dataset = getattr(self.cfg.var.obj_operator, f'{mode}_set')
        
        if mode == 'train':
            for k, v in self.metrics_epoch_train.items():
                self.metrics_epoch_train[k] = v / len(dataset)
        else:
            for k, v in self.metrics_epoch_val.items():
                self.metrics_epoch_val[k] = v / len(dataset)

        if mode == 'test':
            
            if len(self.metrics_samplewise['dice']) > 0:
                self.metrics_samplewise['dice'] = list(torch.concat(self.metrics_samplewise['dice']).cpu().numpy())
                self.metrics_epoch_val['std_dice'] = np.std(self.metrics_samplewise['dice'], ddof=1)
            else:
                del self.metrics_samplewise['dice']
                
            if len(self.metrics_samplewise['assd']) > 0:
                self.metrics_samplewise['assd'] = list(torch.concat(self.metrics_samplewise['assd']).cpu().numpy())
                self.metrics_epoch_val['std_assd'] = np.std(self.metrics_samplewise['assd'], ddof=1)
            else:
                del self.metrics_samplewise['assd']
                
            if len(self.metrics_samplewise['tre']) > 0:
                self.metrics_samplewise['tre'] = list(torch.concat(self.metrics_samplewise['tre']).cpu().numpy())
                self.metrics_epoch_val['std_tre'] = np.std(self.metrics_samplewise['tre'], ddof=1)
            else:
                del self.metrics_samplewise['tre']
            
            if not self.metrics_samplewise['names']:
                del self.metrics_samplewise['names']
                
            df = pd.DataFrame.from_dict(self.metrics_samplewise)
            save_name = 'metrics_samplewise.csv'
            if self.cfg.model.task_sequential:
                if self.cfg.exp.mode == 'train':
                    prefix = self.cfg.dataset.train_tasks[self.cfg.var.obj_operator.task_idx]
                else:
                    prefix = self.cfg.dataset.test_tasks[self.cfg.var.obj_operator.task_idx]
                save_name = prefix + '_' + save_name
            df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, save_name))
            
    def get_metrics(self, data, output, mode='train', replay=False):
        """
            loss_final: used for backward training
            metric_final: used to select best model (higher is better)
            other metrics: for visualization
        """
        self.metrics_iter = OrderedDict(loss_final=0., metric_final=0.)
        if mode in ['train', 'val']:
            if replay:
                self.metrics_replay = OrderedDict(loss_final=0., metric_final=0.)
                for name_loss, w in self.cfg.model.ws_loss.items():
                    if w > 0.:
                        loss = getattr(self, f'loss_{name_loss}')()
                        if torch.is_tensor(loss):
                            self.metrics_replay[f'loss_{name_loss}'] = loss.item()
                            self.metrics_replay['loss_final'] += w * loss
                
                return self.metrics_replay
            else:
                for name_loss, w in self.cfg.model.ws_loss.items():
                    if w > 0.:
                        loss = getattr(self, f'loss_{name_loss}')()
                        if torch.is_tensor(loss):
                            self.metrics_iter[f'loss_{name_loss}'] = loss.item()
                            self.metrics_iter['loss_final'] += w * loss            
                    
        with torch.no_grad():
            disp = self.net.decoder.disp
            
            if 'segs' in self.data:
                segs_warped = self.segs_warped.type(torch.int64)
                
                segs_warped = F.one_hot(segs_warped, num_classes=self.cfg.var.num_classes).float()  # [B, 2, ..., C]
                segs_warped = rearrange(segs_warped, 'B N ... C -> B N C ...')
                
                dices = self.dice(segs_warped[:, 0], segs_warped[:, 1])
                if self.cfg.exp.mode == 'test':
                    assds = monai.metrics.compute_average_surface_distance(segs_warped[:, 0], segs_warped[:, 1], symmetric=True)
                    assds[torch.isnan(assds)] = 0.
                    assds[torch.isinf(assds)] = 0.
                    
                dices_class = torch.mean(dices, dim=0) # [C-1]
                for i, dice in enumerate(dices_class):
                    self.metrics_iter[f'dice_{i + 1}'] = dice
                if self.cfg.exp.mode == 'test':
                    assds_class = torch.mean(assds, dim=0) # [C-1]
                    for i, assd in enumerate(assds_class):
                        self.metrics_iter[f'assd_{i + 1}'] = assd
                        
                self.metrics_iter['dice_mean'] = torch.mean(dices_class)
                if self.cfg.exp.mode == 'test':
                    self.metrics_iter['assd_mean'] = torch.mean(assds_class)
                self.metrics_iter['metric_final'] = self.metrics_iter['dice_mean']
                del segs_warped
                
            if 'keypoints' in self.data:
                tres = torch.stack([torch.square(self.keypoints_warped[i][0] - self.keypoints_warped[i][1]).sum(dim=-1).mean(dim=-1).sqrt() for i in range(self.cfg.var.n_samples)])
                
                self.metrics_iter['tre_mean'] = torch.sum(tres) / torch.sum(tres > 0).clamp(min=1e-8)
                if 'metric_final' not in self.metrics_iter:
                    self.metrics_iter['metric_final'] = self.metrics_iter['tre_mean']
            
            mask_fg = self.fixed_mask
            if mode in ['val', 'test']:
                jacobian = self.jacobian(disp)
                if self.cfg.dataset.dim == 3:
                    jacobian[~mask_fg[:, 0, 1:-1, 1:-1, 1:-1].bool()] = 0
                else:
                    jacobian[~mask_fg[:, 0, 1:-1, 1:-1].bool()] = 0
                
                self.metrics_iter['n_neg_jacobian'] = torch.sum(jacobian < 0) / self.cfg.var.n_samples
                
                if mode == 'test':
                    if 'segs' in self.data:
                        self.metrics_samplewise['dice'].append(torch.mean(dices, dim=1))
                        del dices
                        if self.cfg.exp.mode == 'test':
                            self.metrics_samplewise['assd'].append(torch.mean(assds, dim=1))
                            del assds
                    if 'keypoints' in self.data:
                        self.metrics_samplewise['tre'].append(tres)
                        del tres
                    
                    if 'names' in self.data:
                        self.metrics_samplewise['names'] += self.data['names']
                    
                    if self.cfg.exp.mode == 'test':
                        if self.cfg.exp.test.save_jacobian:
                            path_jacobian = os.path.join(self.cfg.var.obj_operator.path_exp, 'jacobian')
                            if not os.path.exists(path_jacobian):
                                os.makedirs(path_jacobian)
                            for i in range(jacobian.shape[0]):
                                save_image_nii(jacobian[i].cpu().numpy(),
                                            save_path=os.path.join(path_jacobian, f'jacobian_mod{i}.nii.gz'), save_dtype=np.float32)
                        if self.cfg.exp.test.save_result.enable:
                            if self.cfg.exp.test.save_result.idx_sample == -1:
                                bs = list(np.arange(self.imgs_distorted.shape[0]) + self.idx_sample)
                            else:
                                bs = [0]

                            path_result = os.path.join(self.cfg.var.obj_operator.path_exp, 'imgs')
                            if not os.path.exists(path_result):
                                os.makedirs(path_result)
                            
                            for b in bs:
                                b_ = b - self.idx_sample
                                if self.cfg.exp.test.save_result.img_ori:
                                    fixed_img = self.fixed_img[b_, 0].cpu().numpy()
                                    moving_img = self.moving_img[b_, 0].cpu().numpy()
                                    save_image_nii(fixed_img, save_path=os.path.join(path_result, f'fixed_img_b{b}.nii.gz'), save_dtype=np.float32)
                                    save_image_nii(moving_img, save_path=os.path.join(path_result, f'moving_img_b{b}.nii.gz'), save_dtype=np.float32)
                                    
                                if self.cfg.exp.test.save_result.img_warped:
                                    warped_moving_img = self.warped_moving_img[b_, 0].cpu().numpy()
                                    save_image_nii(warped_moving_img, save_path=os.path.join(path_result, f'warped_moving_img_b{b}.nii.gz'), save_dtype=np.float32)
                                    
                                if self.cfg.exp.test.save_result.seg_warped and 'segs' in self.data:
                                    fixed_seg = self.fixed_seg[b_, 0].cpu().numpy()
                                    save_image_nii(fixed_seg, save_path=os.path.join(path_result, f'fixed_seg_b{b}.nii.gz'))
                                    
                                    warped_moving_seg = self.warped_moving_seg[b_, 0].cpu().numpy()
                                    save_image_nii(warped_moving_seg, save_path=os.path.join(path_result, f'warped_moving_seg_b{b}.nii.gz'))
                                    
                                if self.cfg.exp.test.save_result.disp:
                                    array = disp[b_].cpu().numpy()
                                    if self.cfg.dataset.dim == 3:
                                        array = rearrange(array, 'D X Y Z -> X Y Z D')
                                    else:
                                        array = rearrange(array, 'D X Y -> X Y D')
                                        array = array[:, :, None]
                                        np.concatenate([array, np.zeros([*array.shape[:-1], 1])], axis=-1)
                                    
                                    save_image_nii(array, save_path=os.path.join(path_result, f'disp_b{b}.nii.gz'), save_dtype=np.float32)             
                    
                    del jacobian, mask_fg
            
            elif mode != 'train':
                raise ValueError
            self.idx_sample += self.imgs_distorted.shape[0]
            del disp

            for k, v in self.metrics_iter.items():
                if mode == 'train':
                    self.metrics_epoch_train[k] = self.metrics_epoch_train.get(
                    k, 0.) + float(v) * self.cfg.var.n_samples
                else:
                    self.metrics_epoch_val[k] = self.metrics_epoch_val.get(
                    k, 0.) + float(v) * self.cfg.var.n_samples
                            
        return self.metrics_iter

    def vis(self):
        if self.cfg.exp.test.reduce_memory:
            assert self.cfg.exp.mode == 'test'
            return
        self.img_grid = get_img_grid(n_imgs=2,
                                     size_img=self.imgs_distorted.shape[-3:-1])
