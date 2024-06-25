# -*- coding: utf-8 -*-
"""
Modules for computing metrics.

@author: Xinzhe Luo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_segmentation(predictor, mode='torch'):
    """
    produce the segmentation maps from the probability maps

    :param predictor: tensor/array of shape [B, C, *vol_shape]
    """
    assert mode in ['torch', 'np'], "The mode must be either 'torch' or 'np'!"
    if mode == 'torch':
        assert isinstance(predictor, torch.Tensor)
        ndim = predictor.dim()
        return F.one_hot(torch.argmax(predictor, dim=1),
                         predictor.shape[1]).permute(0, -1, *range(1, ndim - 1)).to(torch.float32)

    elif mode == 'np':
        assert isinstance(predictor, np.ndarray)
        ndim = predictor.ndim
        return np.eye(predictor.shape[1])[np.argmax(predictor, axis=1)].transpose((0, -1, *range(1, ndim - 1)))


class OverlapMetrics(nn.Module):
    """
    Compute the Dice similarity coefficient between the ground truth and the prediction.
    Assume the first class is background.

    """
    def __init__(self, eps=1e-5, mode='torch', type='average_foreground_dice', **kwargs):
        super(OverlapMetrics, self).__init__()
        self.eps = eps
        self.mode = mode
        self.type = type
        self.kwargs = kwargs
        self.class_index = kwargs.get('class_index', None)
        self.channel_last = kwargs.get('channel_last', False)

        assert mode in ['torch', 'np'], "The mode must be either 'tf' or 'np'!"
        assert type in ['average_foreground_dice', 'class_specific_dice', 'average_foreground_jaccard', 'all_foreground_dice']

    def forward(self, y_true, y_seg):
        """

        :param y_true: tensor of shape [batch, num_classes, *vol_shape]
        :param y_seg: tensor of shape [batch, num_classes, *vol_shape]
        :return: tensor of shape [batch]
        """
        if self.mode == 'np':
            y_true = torch.from_numpy(y_true)
            y_seg = torch.from_numpy(y_seg)

        dimension = y_true.dim() - 2

        if self.channel_last:
            y_true = y_true.permute(0, -1, *list(range(1, 1 + dimension)))
            y_seg = y_seg.permute(0, -1, *list(range(1, 1 + dimension)))

        assert y_true.size()[1:] == y_seg.size()[1:], "The ground truth and prediction must be of equal shape! " \
                                                      "Ground truth shape: %s, " \
                                                      "prediction shape: %s" % (tuple(y_true.size()),
                                                                                tuple(y_seg.size()))

        n_class = y_seg.size()[1]

        y_seg = get_segmentation(y_seg, mode='torch')

        if self.type in ['average_foreground_dice', 'all_foreground_dice']:
            dice = []
            for i in range(1, n_class):
                top = 2 * torch.sum(y_true[:, i] * y_seg[:, i], dim=tuple(range(1, 1 + dimension)))
                bottom = torch.sum(y_true[:, i] + y_seg[:, i], dim=tuple(range(1, 1 + dimension)))
                dice.append(top.clamp(min=self.eps) / bottom.clamp(min=self.eps))

            metric = torch.stack(dice, dim=-1)  # [B, C]
            if self.type == 'average_foreground_dice':
                metric = metric.mean(dim=1)

        elif self.type == 'class_specific_dice':
            assert self.class_index is not None, "The class index must be provided!"
            top = 2 * torch.sum(y_true[:, self.class_index] * y_seg[:, self.class_index], dim=tuple(
                range(1, 1 + dimension)))
            bottom = torch.sum(y_true[:, self.class_index] + y_seg[:, self.class_index],
                               dim=tuple(range(1, 1 + dimension)))
            metric = top.clamp(min=self.eps) / bottom.clamp(min=self.eps)

        elif self.type == 'average_foreground_jaccard':
            jaccard = []
            y_true = y_true.type(torch.bool)
            y_seg = y_seg.type(torch.bool)
            for i in range(1, n_class):
                top = torch.sum(y_true[:, i] & y_seg[:, i], dtype=torch.float32, dim=tuple(range(1, 1 + dimension)))
                bottom = torch.sum(y_true[:, i] | y_seg[:, i], dtype=torch.float32, dim=tuple(range(1, 1 + dimension)))
                jaccard += top.clamp(min=self.eps) / bottom.clamp(min=self.eps)

            metric = torch.stack(jaccard, dim=1).mean(dim=1)

        else:
            raise ValueError("Unknown overlap metric: %s" % self.type)

        if self.mode == 'np':
            return metric.detach().cpu().numpy()

        return metric
