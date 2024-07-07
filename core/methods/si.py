import torch
import torch.nn as nn
from core.models.continual_reg import ContinualReg


class SI(ContinualReg):
    NAME = 'si'

    def __init__(self, cfg):
        super(SI, self).__init__(cfg)

        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0
        self.cfg = cfg

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.net.get_params().data - self.checkpoint) ** 2 + self.cfg.method.si.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        output = self.forward(inputs)
        self.metrics = self.get_metrics(inputs, output)
        penalty = self.penalty()
        loss = self.metrics['loss_final'] + self.cfg.method.si.c * penalty
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.opt.step()

        self.small_omega += self.cfg.exp.train.optimizer.lr * self.net.get_grads().data ** 2

        return loss.item()
