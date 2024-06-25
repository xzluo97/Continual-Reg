import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track

from core.models.continual_reg import ContinualReg


class EwcOn(ContinualReg):
    NAME = 'ewc_on'
    def __init__(self, cfg):
        super(EwcOn, self).__init__(cfg)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.cfg.var.obj_operator.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, train_loader):
        fish = torch.zeros_like(self.net.get_params())
        total_num_samples = 0
        print(f'----------- {self.NAME} end task -----------')
        for _, data in enumerate(track(train_loader, transient=True, 
                                       description='computing Fisher information matrix')):
            total_num_samples += data['imgs'].shape[0]
            for datum in data['imgs']:
                self.opt.zero_grad()
                inputs = {'imgs': datum.unsqueeze(0).to(self.cfg.var.obj_operator.device)}
                output = self.forward(inputs)
                self.metrics = self.get_metrics(inputs, output)
                loss = self.metrics['loss_final']
                loss.backward()
                fish += self.net.get_grads() ** 2

        fish /= total_num_samples

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.cfg.method.ewc.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    def observe(self, inputs, not_aug_inputs=None):
                
        self.opt.zero_grad()
        output = self.forward(inputs)
        penalty = self.penalty()
        self.metrics = self.get_metrics(inputs, output)
        loss = self.metrics['loss_final'] + self.cfg.method.ewc.e_lambda * penalty
        assert not torch.isnan(loss)
        if self.cfg.exp.train.use_gradscaler:
            self.cfg.var.gradscaler.scale(loss).backward()
            self.cfg.var.gradscaler.unscale_(self.opt)
            self.cfg.var.gradscaler.step(self.opt)
            self.cfg.var.gradscaler.update()
        else:
            loss.backward()
            self.opt.step()

        return output
