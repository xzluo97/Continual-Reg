"""
__author__ = Xinzhe Luo
__version__ = 0.1

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf as Ocfg
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')
from collections import OrderedDict
from einops import rearrange

from core.models.base import Base
from core.networks.LKUNet import LKUNet
from core.losses.NCC import NCC
from core.losses.DiceLoss import DiceLoss
from core.register.LocalDisplacementEnergy import BendingEnergy, MembraneEnergy
from core.models.utils import make_grid, draw_keypoints
from core.data.image_utils import separable_filter2d, separable_filter3d, gauss_kernel1d


class ContinualReg(Base):
    def __init__(self, cfg: Ocfg):
        super().__init__(cfg)
        # self.net = UNet(cfg)
        self.net = LKUNet(cfg)
        
        self.paths_file_net += [
            'core/networks/UNet.py', 
            'core/networks/LKUNet.py',
            'core/models/continual_reg.py'
        ]
        
        self.ncc = NCC(win=cfg.model.ncc.win)
        self.dice_loss = DiceLoss()
        
        self.bending = BendingEnergy(dimension=cfg.dataset.dim)
        self.membrane = MembraneEnergy(dimension=cfg.dataset.dim)
        
        self.opt, self.sch = self._get_optimizer(self.net.parameters())
    
    def loss_ncc(self):
        loss = 0.        
        mask = self.net.decoder.mask

        ncc_idx = range(self.cfg.var.n_samples)
        if self.cfg.net.output_vel and self.cfg.net.symmetric:
            loss += self.ncc.loss(self.net.decoder.half_warped_fixed_img[ncc_idx], 
                                  self.net.decoder.half_warped_moving_img[ncc_idx], 
                                  mask=mask[ncc_idx])
            if self.cfg.model.multiscale:
                for i in range(1, self.cfg.net.n_levels):
                    loss += self.ncc.loss(self.net.decoder.half_warped_fixed_imgs[i][ncc_idx], 
                                          self.net.decoder.half_warped_moving_imgs[i][ncc_idx],
                                          mask=self.net.decoder.masks[i][ncc_idx])
        else:
            loss += self.ncc.loss(self.fixed_img[ncc_idx], 
                                  self.net.decoder.warped_moving_img[ncc_idx], 
                                  mask=mask[ncc_idx])
            if self.cfg.model.multiscale:
                for i in range(1, self.cfg.net.n_levels):
                    loss += self.ncc.loss(self.net.decoder.downsampled_fixed_imgs[i][ncc_idx],
                                          self.net.decoder.warped_moving_imgs[i][ncc_idx],
                                          mask=self.net.decoder.masks[i][ncc_idx])
            
        return loss
    
    def loss_bending(self):
        if self.cfg.model.regularization.over == 'vel' and self.cfg.net.output_vel:
            loss = self.bending(self.net.decoder.vel)
            if self.cfg.model.multiscale:
                for i in range(1, self.cfg.net.n_levels):
                    loss += self.bending(self.net.decoder.vels[i])
        else:
            loss = self.bending(self.net.decoder.disp)
            if self.cfg.model.multiscale:
                for i in range(1, self.cfg.net.n_levels):
                    loss += self.bending(self.net.decoder.disps[i])
            
        return loss
    
    def loss_membrane(self):
        if self.cfg.model.regularization.over == 'vel' and self.cfg.net.output_vel:
            loss = self.membrane(self.net.decoder.vel)
            if self.cfg.model.multiscale:
                for i in range(1, self.cfg.net.n_levels):
                    loss += self.membrane(self.net.decoder.vels[i])
        else:
            loss = self.membrane(self.net.decoder.disp)
            if self.cfg.model.multiscale:
                for i in range(1, self.cfg.net.n_levels):
                    loss += self.membrane(self.net.decoder.disps[i])
            
        return loss
    
    def loss_dice(self):
        loss = 0.
        
        if self.cfg.dataset.dim == 3:
            spatial_filter = separable_filter3d
            interp_mode = 'trilinear'
        else:
            spatial_filter = separable_filter2d
            interp_mode = 'bilinear'
        
        if 'segs' in self.data:
            segs_idx = list(range(len(self.cfg.var.dataset_names)))
            if len(segs_idx) > 0:
                segs_distorted = F.one_hot(self.segs_distorted[segs_idx].long(), num_classes=self.cfg.var.num_classes).float()
                segs_distorted = rearrange(segs_distorted, 'B N ... C -> B (N C) ...')
                probs_distorted = spatial_filter(segs_distorted, 
                                                 kernel=gauss_kernel1d(self.cfg.model.dice.kernel_sigma))
                probs_distorted = rearrange(probs_distorted, 'B (N C) ... -> B N C ...', C=self.cfg.var.num_classes)

                if self.cfg.net.output_vel and self.cfg.net.symmetric:
                    warped_fixed_prob = self.net.decoder.warpers[0](probs_distorted[:, 0], 
                                                                    self.net.decoder.half_inv_disp[segs_idx])
                    warped_moving_prob = self.net.decoder.warpers[0](probs_distorted[:, 1], 
                                                                    self.net.decoder.half_disp[segs_idx])
                    loss += self.dice_loss(warped_fixed_prob, warped_moving_prob)
                else:
                    warped_moving_prob = self.net.decoder.warpers[0](probs_distorted[:, 1], 
                                                                    self.net.decoder.disp[segs_idx])
                    loss += self.dice_loss(probs_distorted[:, 0], warped_moving_prob)
                
                if self.cfg.model.multiscale:
                    probs_distorted_ori = rearrange(probs_distorted, 'B N C ... -> B (N C) ...')
                    for i in range(1, self.cfg.net.n_levels):
                        probs_distorted = F.interpolate(probs_distorted_ori, scale_factor=2 ** (-i), mode=interp_mode)
                        probs_distorted = rearrange(probs_distorted, 'B (N C) ... -> B N C ...', C=self.cfg.var.num_classes)
                        if self.cfg.net.output_vel and self.cfg.net.symmetric:
                            warped_fixed_prob = self.net.decoder.warpers[i](probs_distorted[:, 0], self.net.decoder.half_inv_disps[i][segs_idx])
                            warped_moving_prob = self.net.decoder.warpers[i](probs_distorted[:, 1], self.net.decoder.half_disps[i][segs_idx])
                            loss += self.dice_loss(warped_fixed_prob, warped_moving_prob)
                        else:
                            warped_moving_prob = self.net.decoder.warpers[0](probs_distorted[:, 1], 
                                                                            self.net.decoder.disps[i][segs_idx])
                            loss += self.dice_loss(probs_distorted[:, 0], warped_moving_prob)
                        
        return loss
    
    def loss_tre(self):
        loss = 0.
        
        if 'keypoints' in self.data:
            if self.cfg.model.tre.label_center:
                keypoints_idx = list(range(len(self.cfg.var.dataset_names)))
            else:
                keypoints_idx = [i for i in range(len(self.cfg.var.dataset_names)) 
                                 if self.cfg.var.dataset_names[i] == 'NLST']
            if len(keypoints_idx) > 0:
                tres = torch.stack([torch.square(self.keypoints_warped[i][0] - self.keypoints_warped[i][1]).sum(dim=-1).mean(dim=-1).sqrt() for i in keypoints_idx])
                
                loss += torch.mean(tres)
            
        return loss          
    
    def observe(self, inputs, not_aug_inputs):
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError
        
    def vis(self, writer, global_step, data, output, mode, in_epoch):
        if in_epoch:
            return
        if self.cfg.exp.test.reduce_memory:
            return

        super().vis()
        with torch.no_grad():
            if mode in ['val', 'test']:
                b = 0
                dataset_name = self.cfg.var.dataset_names[b]
                if dataset_name == 'OASIS':
                    # imgs_show: [2, 1, H, W]
                    z = self.cfg.var.img_size[0] // 2
                    imgs_show = OrderedDict()
                    imgs_show['imgs distorted'] = self.imgs_distorted[b, :, z].unsqueeze(1)
                    imgs_show['imgs registered'] = self.imgs_warped[b, :, z].unsqueeze(1)
                    imgs_show['segs distorted'] = self.segs_distorted[b, :, z].unsqueeze(1) / self.segs_distorted[b].amax()
                    imgs_show['segs registered'] = self.segs_warped[b, :, z].unsqueeze(1) / self.segs_warped[b].amax()
                    cmap = 'gray'
                else:
                    z = self.cfg.var.img_size[-1] // 2
                    imgs_show = OrderedDict()
                    imgs_distorted = self.imgs_distorted[b, ..., z].unsqueeze(1)
                    imgs_registered = self.imgs_warped[b, ..., z].unsqueeze(1)
                    if 'segs' in self.data:
                        imgs_show['segs distorted'] = self.segs_distorted[b, ..., z].unsqueeze(1) / self.segs_distorted[b].amax()
                        imgs_show['segs registered'] = self.segs_warped[b, ..., z].unsqueeze(1) / self.segs_warped[b].amax()
                    
                    if dataset_name == 'NLST':
                        kpts_distorted = torch.cat([self.fixed_kpt[b], 
                                                     self.fixed_kpt[b]]).div(2) # [2, n, 3]
                        kpts_registered = kpts_distorted.clone()
                                                
                        imgs_distorted_lst = []
                        imgs_registered_lst = []
                        for i in range(2):
                            kpt_dis = kpts_distorted[[i]] # [1, n, 3]
                            kpt_dis = kpt_dis[kpt_dis[..., [-1]].round().repeat(1, 1, self.cfg.dataset.dim) == z].view(1, -1, self.cfg.dataset.dim)[..., :-1]
                            img_dis_kpt = draw_keypoints(imgs_distorted[i].mul(255).repeat(3, 1, 1).to(torch.uint8), 
                                                         keypoints=kpt_dis, colors='purple', radius=1)
                            imgs_distorted_lst.append(img_dis_kpt)
                            
                            kpt_reg = kpts_registered[[i]] # [1, n, 3]
                            kpt_reg = kpt_reg[kpt_reg[..., [-1]].round().repeat(1, 1, self.cfg.dataset.dim) == z].view(1, -1, self.cfg.dataset.dim)[..., :-1]
                            img_reg_kpt = draw_keypoints(imgs_registered[i].mul(255).repeat(3, 1, 1).to(torch.uint8),
                                                         keypoints=kpt_reg, colors='purple', radius=1)
                            imgs_registered_lst.append(img_reg_kpt)
                            
                        imgs_show['imgs distorted'] = torch.stack(imgs_distorted_lst)
                        imgs_show['imgs registered'] = torch.stack(imgs_registered_lst) # [2, 3, H, W]
                        imgs_show['segs distorted'] *= 255
                        imgs_show['segs registered'] *= 255
                        cmap = None
                    else:
                        imgs_show['imgs distorted'] = imgs_distorted
                        imgs_show['imgs registered'] = imgs_registered
                        cmap = 'gray'
                
                img_show_size = imgs_show['imgs distorted'].shape[-2:]
                rows = 2 if 'segs' in self.data else 1
                w_fig = 0.42 * 2 * img_show_size[1] / 20
                h_fig = img_show_size[0] / 25 + rows * 0.25
                w_fig *= 2
                h_fig = (h_fig - rows * 0.25) * 2 + 3 * 0.25
                fig, axes = plt.subplots(rows, 2, figsize=(w_fig, h_fig))
                for ax, key in zip(axes.reshape(-1), imgs_show.keys()):
                    ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                    if key == 'displacements':
                        pad_value = 0.
                    else:
                        pad_value = 255.
                    imgs = imgs_show[key]
                    if dataset_name != 'NLST':
                        imgs *= 255
                    ax.imshow(make_grid(imgs, nrow=2, pad_value=pad_value).cpu().numpy()[0], cmap=cmap)
                    ax.set_title(key, fontsize='14')
                fig.tight_layout(pad=0.01)
                writer.add_figure(mode, fig, global_step)
                if self.cfg.var.obj_operator.is_best:
                    writer.add_figure(f'{mode}_best', fig, global_step)
                fig.clf()

            elif mode != 'train':
                raise ValueError

    def _get_optimizer(self, params):
        name_opt = self.cfg.exp.train.optimizer.name
        cfg_opt = self.cfg.exp.train.optimizer[name_opt]

        if name_opt == 'sgd':
            optimizer = optim.SGD(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
                momentum=cfg_opt.momentum,
                nesterov=cfg_opt.nesterov,
            )
        elif name_opt in ('adam', 'adamw'):
            optimizer = optim.Adam(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
            )
        else:
            raise NotImplementedError(f'Unknown optimizer: {name_opt}')

        name_sch = self.cfg.exp.train.scheduler.name
        if name_sch is not None:
            cfg_sch = self.cfg.exp.train.scheduler[name_sch]
            name_sch = name_sch.lower()
            if name_sch == 'cycliclr':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=cfg_sch.lr_base,
                                                        max_lr=cfg_sch.lr_max, mode=cfg_sch.mode, gamma=cfg_sch.gamma,
                                                        cycle_momentum=cfg_sch.cycle_momentum)
            elif name_sch == 'multisteplr':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg_sch.milestones,
                                                           gamma=cfg_sch.gamma)
            elif name_sch == 'exponentiallr':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg_sch.gamma)
            elif name_sch == 'lambdalr':
                lambdas_lr = []
                for where in cfg_sch.where: # e.g., model.lambda_lr_0
                    attrs = where.split('.')
                    lambda_lr = self
                    for attr in attrs:
                        lambda_lr = getattr(lambda_lr, attr)
                    lambdas_lr.append(lambda_lr)
                scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambdas_lr)
            else:
                raise NotImplementedError(f'Unknown scheduler: {name_sch}')
        else:
            scheduler = None

        return optimizer, scheduler