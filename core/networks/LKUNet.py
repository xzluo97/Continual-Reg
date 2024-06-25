from core.register.SpatialTransformer import SpatialTransformer, ResizeTransform
from core.register.VectorIntegration import VectorIntegration
import torch
import torch.nn as nn


class LKConvBlk(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()
        self.cfg = cfg
        self.output = None
        
        if cfg.dataset.dim == 2:
            Conv = nn.Conv2d
        elif cfg.dataset.dim == 3:
            Conv = nn.Conv3d
        else:
            raise NotImplementedError
        
        self.activation = nn.PReLU()
        
        self.conv_regular = Conv(in_channels, out_channels, kernel_size=3, padding='same', bias=True)
        
        self.conv_one = Conv(in_channels, out_channels, kernel_size=1, bias=True)
        
        self.conv_large = Conv(in_channels, out_channels, kernel_size=cfg.net.large_kernel, padding='same')
        
    def forward(self, x):
        
        y = self.conv_regular(x) + self.conv_one(x) + self.conv_large(x) + x
        
        return self.activation(y)
    
    
class LKEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if cfg.dataset.dim == 2:
            Conv = nn.Conv2d
        elif cfg.dataset.dim == 3:
            Conv = nn.Conv3d
        else:
            raise NotImplementedError
        
        self.init_conv = nn.Sequential(Conv(cfg.dataset.n_channels_img * 2, 
                                            cfg.net.n_channels_init, 
                                            kernel_size=3, padding='same'), 
                                       nn.PReLU(), 
                                       Conv(cfg.net.n_channels_init,
                                            cfg.net.n_channels_init,
                                            kernel_size=3, padding='same'),
                                       nn.PReLU())
        
        
        self.conv_blks = nn.ModuleList()
        self.LKconv_blks = nn.ModuleList()
        for i in range(cfg.net.n_levels):
            in_channels_down = cfg.net.n_channels_init * 2 ** i

            if i < cfg.net.n_levels - 1:
                out_channels_down = in_channels_down * 2
            else:
                out_channels_down = out_channels_down
            
            self.conv_blks.append(
                nn.Sequential(Conv(in_channels_down, out_channels_down, 
                                   kernel_size=3, stride=2, padding=1), 
                              nn.PReLU())
                )

            self.LKconv_blks.append(
                LKConvBlk(cfg, out_channels_down, out_channels_down)
                )
            
    def forward(self, x):
        x = self.init_conv(x)
        output = [x]
        
        for i in range(self.cfg.net.n_levels):
            x = self.conv_blks[i](x)
            output.append(x)
            x = self.LKconv_blks[i](x)
            
        output.append(x)

        return output
    
    
class LKDecoder(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if cfg.dataset.dim == 2:
            Conv = nn.Conv2d
            Deconv = nn.ConvTranspose2d
        elif cfg.dataset.dim == 3:
            Conv = nn.Conv3d
            Deconv = nn.ConvTranspose3d
        else:
            raise NotImplementedError
        
        self.warpers = nn.ModuleList([SpatialTransformer([s // 2 ** i for s in cfg.dataset.size_img]) for i in range(cfg.net.n_levels)])
        self.vel2disps = nn.ModuleList([VectorIntegration([s // 2 ** i for s in cfg.dataset.size_img], int_steps=7) for i in range(cfg.net.n_levels)])
        self.downsamples = nn.ModuleList([nn.Upsample(scale_factor=2 ** (-i),
                                                      mode='trilinear' if cfg.dataset.dim == 3 else 'bilinear')
                                          for i in range(cfg.net.n_levels)])
        
        self.conv_blks = nn.ModuleList()
        self.deconv_blks = nn.ModuleList()
        
        for i in range(cfg.net.n_levels):
            in_channels_up = cfg.net.n_channels_init * 2 ** i
            
            self.deconv_blks.append(
                nn.Sequential(Deconv(in_channels_up, in_channels_up, 
                                     kernel_size=2, stride=2),
                              nn.PReLU())
            )
            
            self.conv_blks.append(
                nn.Sequential(Conv(in_channels_up * 2, in_channels_up, 
                                   kernel_size=3, padding='same'),
                              nn.PReLU(),
                              Conv(in_channels_up, 
                                   in_channels_up if i == 0 else in_channels_up // 2 , 
                                   kernel_size=3, padding='same'), 
                              nn.PReLU()
                              )
            )
            
        self.out_conv = Conv(cfg.net.n_channels_init, cfg.dataset.dim, 
                             kernel_size=3, padding='same', bias=False)
        nn.init.zeros_(self.out_conv.weight)
        
    def forward(self, out):
        
        x = out[-1]
        
        for i in range(self.cfg.net.n_levels - 1, -1, -1):
            x = torch.cat([self.deconv_blks[i](x), out[i]], dim=1)
            x = self.conv_blks[i](x)
            
        y = self.out_conv(x)
        
        if self.cfg.net.output_vel:
            self.vel = y
            if self.cfg.net.symmetric:
                self.half_disp = self.vel2disps[0](self.vel)
                self.half_inv_disp = self.vel2disps[0](- self.vel)
                self.disp = self.warpers[0].compose_flows(flows=[self.half_disp, self.half_disp])
            else:
                self.disp = self.vel2disps[0](self.vel)
        else:
            self.disp = y
                        
        self.warped_moving_img = self.warpers[0](self.cfg.var.obj_model.moving_img, 
                                                 self.disp) # [B, 1, ...]
        
        if self.cfg.net.output_vel and self.cfg.net.symmetric:
            self.half_warped_fixed_img = self.warpers[0](self.cfg.var.obj_model.fixed_img, 
                                                         self.half_inv_disp)
            self.half_warped_moving_img = self.warpers[0](self.cfg.var.obj_model.moving_img, 
                                                          self.half_disp)
            
            warped_moving_mask = self.warpers[0](self.cfg.var.obj_model.moving_mask,
                                                 self.half_disp, interp_mode='nearest')
            overlap_mask = self.warpers[0].getOverlapMask(self.half_disp)
            mask = torch.logical_and(warped_moving_mask, overlap_mask)
            warped_fixed_mask = self.warpers[0](self.cfg.var.obj_model.fixed_mask, 
                                                self.half_inv_disp, 
                                                interp_mode='nearest')
            mask = torch.logical_and(mask, warped_fixed_mask)
            overlap_mask = self.warpers[0].getOverlapMask(self.half_inv_disp)
            mask = torch.logical_and(mask, overlap_mask)
        
        else:
            mask = self.warpers[0](self.cfg.var.obj_model.moving_mask,  
                                   self.disp, interp_mode='nearest')
            overlap_mask = self.warpers[0].getOverlapMask(self.disp)
            mask = torch.logical_and(mask, overlap_mask)
            mask = torch.logical_and(mask, self.cfg.var.obj_model.fixed_mask)
            
        self.mask = mask
        
        return self.disp
    
    
    
class LKUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = LKEncoder(cfg)
        self.decoder = LKDecoder(cfg)

    def forward(self, x):
        # x: [B, 2, ...]
        out = self.encoder(x)
        disp = self.decoder(out)
        return disp

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for name, pp in self.named_parameters():
            if pp.grad is None:
                print(f"Parameter with name {name} has no gradient!")
                raise ValueError
            else:
                grads.append(pp.grad.view(-1))
        return grads
