import torch
from core.methods.buffer_mer import Buffer
from core.models.continual_reg import ContinualReg


class MER(ContinualReg):
    NAME = 'MER'

    def __init__(self, cfg):
        super(MER, self).__init__(cfg)
        self.cfg = cfg
        self.buffer = Buffer(cfg)

    def draw_batches(self, inputs):
        batches = []
        # batch_num = 1
        for _ in range(1):
            if not self.buffer.is_empty():
                buf_inputs = self.buffer.get_data_dict()
                batches.append(buf_inputs)
            else:
                batches.append(inputs)
        return batches

    def observe(self, inputs, not_aug_inputs=None):
        
        theta_A0 = self.net.get_params().data.clone()
        
        self.opt.zero_grad()
        output = self.forward(inputs)
        self.metrics = self.get_metrics(inputs, output)
        loss = self.metrics['loss_final']
        loss.backward()
        self.opt.step()
            
        self.buffer.add_data_dict(inputs)
        batches = self.draw_batches(inputs)
        
        self.opt.zero_grad()
        output = self.forward(batches[0])
        metrics = self.get_metrics(batches[0], output, replay=True)
        loss = metrics['loss_final']
        loss.backward()
        self.opt.step()
        
        new_params = theta_A0 + self.cfg.method.mer.beta * (self.net.get_params() - theta_A0)
        self.net.set_params(new_params)

        return output
