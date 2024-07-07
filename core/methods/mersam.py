import torch
from core.methods.buffer_mer import Buffer
from core.models.continual_reg import ContinualReg
from core.methods.sam import SAM


class MERSAM(ContinualReg):
    NAME = 'MERSAM'

    def __init__(self, cfg):
        super(MERSAM, self).__init__(cfg)
        self.cfg = cfg
        self.buffer = Buffer(cfg)
        self.opt_sam = SAM(self.net.parameters(),
                           base_optimizer=torch.optim.Adam, 
                           lr=cfg.exp.train.optimizer.lr,
                           rho=cfg.method.sam.rho, 
                           weight_decay=cfg.method.sam.weight_decay,
                           adaptive=cfg.method.sam.adaptive)

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

        batches = self.draw_batches(inputs)
        theta_A0 = self.net.get_params().data.clone()

        for i in range(self.cfg.exp.train.batch_size):
            sample_input = {}
            for k, v in inputs.items():
                if k in ['keypoints', 'names']:
                    sample_input.update({k: [v[i]]})
                else:
                    sample_input.update({k: v[[i]]})

            # within-batch step
            output = self.forward(sample_input)
            metrics = self.get_metrics(sample_input, output, replay=True)
            loss = metrics['loss_final']
            loss.backward()
            self.opt_sam.first_step(zero_grad=True)

            # second step of sam
            output_second = self.forward(sample_input)
            metrics_second = self.get_metrics(sample_input, output_second, replay=True)
            loss = metrics_second['loss_final']
            loss.backward()
            self.opt_sam.second_step(zero_grad=True)

        self.buffer.add_data_dict(inputs)
        batches = self.draw_batches(inputs)
        
        output = self.forward(batches[0])
        self.metrics = self.get_metrics(batches[0], output)
        loss = self.metrics['loss_final']
        loss.backward()
        self.opt_sam.first_step(zero_grad=True)

        # second step of sam
        output_second = self.forward(batches[0])
        metrics_second = self.get_metrics(batches[0], output_second, replay=True)
        loss = metrics_second['loss_final']
        loss.backward()
        self.opt_sam.second_step(zero_grad=True)
        
        # within batch reptile meta-update
        new_new_params = theta_A0 + self.cfg.method.mere.beta * (self.net.get_params() - theta_A0)
        self.net.set_params(new_new_params)

        return output