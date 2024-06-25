from core.models.continual_reg import ContinualReg
import torch.nn as nn


class Sgd(ContinualReg):
    NAME = 'sgd'

    def __init__(self, cfg):
        super(Sgd, self).__init__(cfg)

    def observe(self, inputs, not_aug_inputs=None):
        self.opt.zero_grad()
        output = self.forward(inputs)
        self.metrics = self.get_metrics(inputs, output)
        loss = self.metrics['loss_final']
        loss.backward()
        # nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        return output
