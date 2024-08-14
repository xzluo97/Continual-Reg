import numpy as np
import torch


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_seen_examples = 0
        self.device = 'cpu' if cfg.method.buffer.cpu else cfg.var.obj_operator.device

    def __len__(self):
        return min(self.num_seen_examples, self.cfg.method.er.buffer_size)
    
    def reservoir(self, num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1

    def init_buffer_list(self):
        return setattr(self, 'buffer_data', 
                       [{} for _ in range(self.cfg.method.er.buffer_size)])

    def add_data_dict(self, data_dict):
        if not hasattr(self, 'buffer_data'):
            self.init_buffer_list()

        for j in range(self.cfg.exp.train.batch_size):
            index = self.reservoir(self.num_seen_examples, self.cfg.method.er.buffer_size)
            self.num_seen_examples += 1
            for k, v in data_dict.items():
                if k in ['keypoints', 'names']:
                    self.buffer_data[index].update({k: v[j]})
                else:
                    self.buffer_data[index].update({k: v[j].to(self.device)})
                        
    def get_data_dict(self):
        choice = np.random.choice(min(self.num_seen_examples, len(self.buffer_data)),
                                  size=self.cfg.exp.train.batch_size, replace=False)

        selected_data = {'imgs': [], 'masks': [], 'segs': [], 
                         'keypoints': [], 'names': []}
        
        for k in selected_data.keys():
            for idx in choice:
                selected_data[k].append(self.buffer_data[idx][k])
            
            if k not in ['keypoints', 'names']:
                selected_data[k] = torch.stack(selected_data[k]).to(self.cfg.var.obj_operator.device)
            
        return selected_data

    def is_empty(self) -> bool:
        if self.num_seen_examples == 0:
            return True
        else:
            return False
