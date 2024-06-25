import numpy as np
import torch
from core.models.continual_reg import ContinualReg
import torch.nn.functional as F
from rich.progress import track


class GPM(ContinualReg):
    NAME = 'GPM'

    def __init__(self, cfg):
        super(GPM, self).__init__(cfg)
        self.cfg = cfg
        self.current_task = 0
        self.feature_list = []
        self.sin_value_list = []
        self.eps = 1e-8

    def layer_gradient_cl_reduced_conv(self, output, inputs, device, allow_unused=True, create_graph=False):
        '''
        Compute the layer-wise gradient given loss and parameters`
        '''
        assert output.dim() == 0

        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)

        layer_gradient_projection = []
        for i, inp in enumerate(inputs):
            #print("i", i, inp.size())
            if inp.dim() == self.cfg.dataset.dim + 2:
                [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
                grad = torch.zeros_like(inp) if grad is None else grad
                
                grad = grad.view(grad.shape[0], -1)
                grad_np = grad.detach().cpu().numpy() # [out_channels, in_channels, kT, kH, kW]
                layer_gradient_projection.append(grad_np)
        return layer_gradient_projection
    
    def get_layer_gradient_for_each_layer_reduced_conv(self, train_loader):
        '''
        Get gradient matrix for each layer`
        '''
        self.net.train()
        all_gradients = []
        
        if self.cfg.method.gpm.num_samples:
            total = self.cfg.method.gpm.num_samples // self.cfg.exp.train.batch_size
        else:
            total = len(train_loader) // self.cfg.exp.train.batch_size
        for i, data in enumerate(track(train_loader, transient=True, total=total, description='computing layer-wise gradients')):
            if i < total:
                for j in range(self.cfg.exp.train.batch_size):
                    inputs = {'imgs': data['imgs'][[j]].to(self.cfg.var.obj_operator.device)}
                    if 'segs' in data:
                        inputs['segs'] = data['segs'][[j]].to(self.cfg.var.obj_operator.device)
                    if 'masks' in data:
                        inputs['masks'] = data['masks'][[j]].to(self.cfg.var.obj_operator.device)
                    if 'keypoints' in data:
                        inputs['keypoints'] = [data['keypoints'][j].to(self.cfg.var.obj_operator.device)]
                    if 'names' in data:
                        inputs['names'] = [data['names'][j]]
                    output = self.forward(inputs)
                    self.metrics = self.get_metrics(inputs, output, replay=True)
                    loss = self.metrics['loss_final']
                    
                    layer_gradient_projection = self.layer_gradient_cl_reduced_conv(loss, self.net.parameters(), device=self.cfg.var.obj_operator.device)
                    
                    all_gradients.append(layer_gradient_projection)

        summarized_g = []
        for j in range(len(layer_gradient_projection)):
            layer_g = []
            for i in range(len(all_gradients)):
                layer_g.append(all_gradients[i][j])

            layer_g = np.array(layer_g)  # [N, out_channels, in_channels*kT*kH*kW]
            layer_g = np.reshape(layer_g, (-1, layer_g.shape[2]))

            layer_g = np.transpose(layer_g)  # [in_channels*kT*kH*kW, N*out_channels]
            summarized_g.append(layer_g)

        return summarized_g
    
    def update_GPM(self, mat_list, threshold):
        '''
        Update GPM matrix
        '''
        print('Threshold: ', threshold)
        print("len matlist", len(mat_list))
        if not self.feature_list:
            # After First Task
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + self.eps)
                r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
                self.feature_list.append(U[:, 0:r])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()

                act_hat = activation - self.feature_list[i] @ self.feature_list[i].transpose() @ activation
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + self.eps)
                accumulated_sval = (sval_total - sval_hat) / (sval_total + self.eps)

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating GPM for layer: {}'.format(i + 1))
                    continue
                # update GPM
                Ui = np.hstack((self.feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    self.feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    self.feature_list[i] = Ui

        return self.feature_list

    def begin_task(self, train_loader):

        if self.current_task == 0:
            pass
        else:
            # logger.info('Generate projection matrix for each layer')
            self.feature_mat = []

            for k in range(len(self.feature_list)):
                Uf = torch.as_tensor(self.feature_list[k] @ self.feature_list[k].transpose(), device=self.cfg.var.obj_operator.device)

                self.feature_mat.append(Uf) # [in_channels*kT*kH*kW, in_channels*kT*kH*kW]

    def end_task(self, train_loader):
        self.current_task += 1
        
        print('-' * 11 + f' task-{self.cfg.var.obj_operator.task_idx} GPM update begins ' + '-' * 11)
        mat_list = self.get_layer_gradient_for_each_layer_reduced_conv(train_loader)
        torch.cuda.empty_cache()

        threshold = [self.cfg.method.gpm.threshold + self.current_task * self.cfg.method.gpm.step_size] * len(mat_list)
        self.update_GPM(mat_list, threshold)


    def observe(self, inputs, not_aug_inputs=None):
        # now compute the grad on the current data
        self.opt.zero_grad()
        output = self.forward(inputs)
        self.metrics = self.get_metrics(inputs, output)
        loss = self.metrics['loss_final']
        loss.backward()

        # check if gradient violates buffer constraints
        if self.current_task == 0:
            pass
        else:
            kk = 0

            for k, (m, params) in enumerate(self.net.named_parameters()):
                if params.dim() == self.cfg.dataset.dim + 2:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), self.feature_mat[kk]).view(params.size())
                    kk += 1

        self.opt.step()

        return output