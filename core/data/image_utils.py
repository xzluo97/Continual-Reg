# -*- coding: utf-8 -*-
"""
Data utility functions for image loader and processing.

@author: Xinzhe Luo
@version: 0.1
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import itertools
import re
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import rescale
from scipy import stats
from sklearn import mixture
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import math
from scipy import signal
from einops import rearrange


def get_img_grid(n_imgs=1, size_img=(224, 224), padding_min=4, thickness=1, spacing=9):
    """Get images of white grids. 

    Args:
        n_imgs:
        size_img: Height and width of the image.
        padding_min: Minimum padding.
        thickness: Thickness of each line.
        spacing: Spacing between two adjacent lines.
    Returns:
        img_grid: Shape of [n_imgs, 1, *size_img]. img_grid[i] is identical for i.
    """
    img_grid = torch.zeros(n_imgs, 1, *size_img)

    for dim in [0, 1]:
        n_lines = np.floor((size_img[dim] - 2 * padding_min - thickness) / (thickness + spacing)) + 1
        padding = (size_img[dim] - ((n_lines - 1) * (thickness + spacing) + thickness)) // 2
        idxs = np.arange(n_lines) * (thickness + spacing) + padding
        idxs = idxs.astype(int)

        for i in range(thickness):
            if dim == 0:
                img_grid[..., idxs + i, :] = 1
            else:
                img_grid[..., :, idxs + i] = 1
    return img_grid


#################### Functions to sort filenames into human order ####################


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def strsort(alist):
    alist.sort(key=natural_keys)
    return alist


def load_image_nii(path, dtype=np.float32, scale=0, order=1):
    img = nib.load(path)
    image = np.asarray(img.get_fdata(), dtype)
    if scale > 0:
        image = rescale(image, 1 / (2**scale), mode='reflect', multichannel=False, anti_aliasing=False, order=order)
    return image, img.affine, img.header


def save_image_nii(array, save_path, **kwargs):
    affine = kwargs.pop("affine", np.eye(4))
    header = kwargs.pop("header", None)
    save_dtype = kwargs.pop("save_dtype", np.uint16)
    img = nib.Nifti1Image(np.asarray(array, dtype=save_dtype), affine=affine, header=header)
    nib.save(img, save_path)


def load_image_png(path, dtype=np.float32, scale=0, order=1):
    img = Image.open(path)
    image = np.asarray(img, dtype)
    if scale > 0:
        image = rescale(image, 1 / (2**scale), mode='reflect', multichannel=False, anti_aliasing=False, order=order)
    return image


def save_image_png(array, save_path, **kwargs):
    normalize = kwargs.pop('normalize', True)
    if normalize:
        array = normalize_image(array, 'min-max')
    img = Image.fromarray(np.asarray(array * 255, dtype=np.uint8))
    img.save(save_path)

def load_prob_file(path_list, dtype=np.float32, max_value=1000, scale=0):
    return np.asarray(
        np.stack([load_image_nii(name, order=0, scale=scale)[0] for name in path_list], -1) / max_value, dtype=dtype)


def normalize_image(image, normalization=None, **kwargs):
    if normalization == 'min-max':
        image -= np.min(image)
        image /= np.max(image)

    elif normalization == 'z-score':
        image = stats.zscore(image, axis=None, ddof=1)
        if kwargs.pop('clip_value', None):
            image = np.clip(image, -3, 3)

    elif normalization == 'interval':
        image -= np.min(image)
        image /= np.max(image)
        a = kwargs.pop('a', -1)
        b = kwargs.pop('b', 1)
        image = (b - a) * image + a

    return image


def get_foreground_center(label, label_intensities):
    """
    Compute the center coordinates of the label according to the given label intensities.

    :param label: The label to derive center coordinates.
    :param label_intensities: A list of intensity values regarded as foreground.
    :return: An array representing the foreground center coordinates, of shape [3].
    """
    foreground_flag = np.any(
        np.concatenate(tuple([(np.expand_dims(label, axis=0) == k) for k in label_intensities[1:]])), axis=0)
    return np.floor(np.mean(np.stack(np.where(foreground_flag)), -1)).astype(np.int16)


def get_label_center(label):
    """
    Compute the center coordinates of the label.

    :param label: tensor of shape [N, *vol_shape].
    :return: tensor of shape [N, C, d].
    """
    label = label.long()
    num_classes = torch.amax(label).item() + 1
    
    centers = []
    for lab in label:
        center = []
        for i in range(1, num_classes):
            coords = torch.nonzero(lab == i)
            if torch.any(coords):
                c = torch.mean(coords.float(), dim=0).round()
            else:
                c = torch.zeros([lab.ndim])
            center.append(c)
            
        centers.append(torch.stack(center))

    return torch.stack(centers)


def get_roi_coordinates(label, label_intensities, mag_rate=0.1):
    """
    Produce the cuboid ROI coordinates representing the opposite vertices.

    :param label: A ground-truth label image.
    :param label_intensities: A list of intensity values regarded as foreground.
    :param mag_rate: The magnification rate for ROI cropping.
    :return: An array representing the smallest coordinates of ROI;
        an array representing the largest coordinates of ROI.
    """

    foreground_flag = np.any(
        np.concatenate(tuple([(np.expand_dims(label, axis=0) == k) for k in label_intensities[1:]])), axis=0)
    arg_index = np.argwhere(foreground_flag)

    low = np.min(arg_index, axis=0)
    high = np.max(arg_index, axis=0)

    soft_low = np.maximum(np.floor(low - (high - low) * mag_rate / 2), np.zeros_like(low))
    soft_high = np.minimum(np.floor(high + (high - low) * mag_rate / 2), np.asarray(label.shape) - 1)

    return soft_low, soft_high


def get_mixture_coefficients(image, label, num_subtypes, channel_first=False):
    """
    Get the image mixture coefficients of each subtype within the tissue class.

    :param image: The image array of shape [*vol_shape, channels].
    :param label: The one-hot label array of shape [*vol_shape, num_classes].
    :param num_subtypes: A list of numbers for each class of labels.
    :return: tau - a list of arrays of shape [num_subtypes[i]];
             mu - a list of arrays of shape [num_subtypes[i]];
             sigma - a list of arrays of shape [num_subtypes[i]].
    """
    num_classes = len(num_subtypes)
    tau = []
    mu = []
    sigma = []
    for i in range(num_classes):
        if channel_first:
            image_take = np.take(np.sum(image, axis=0).flatten(), indices=np.where(label[i].flatten() == 1))
        else:
            image_take = np.take(np.sum(image, axis=-1).flatten(), indices=np.where(label[..., i].flatten() == 1))
        clf = mixture.GaussianMixture(n_components=num_subtypes[i])
        clf.fit(image_take.reshape(-1, 1))
        tau.append(clf.weights_)
        mu.append(clf.means_.squeeze(1))
        sigma.append(np.sqrt(clf.covariances_.squeeze((1, 2))))
    return tau, mu, sigma


def sample_mixture_image(tau, mu, sigma, label, channel_first=False):
    """
    Sample an image from the given Gaussian mixture model.

    :param tau: a list of arrays of shape [num_subtypes[i]], of length num_classes
    :param mu: a list of arrays of shape [num_subtypes[i]], of length num_classes
    :param sigma: a list of arrays of shape [num_subtypes[i]], of length num_classes
    :param label: The one-hot label array of shape [*vol_shape, num_classes]
    :return: sampled image of shape [*vol_shape]
    """
    if channel_first:
        vol_shape = label.shape[1:]
        num_classes = label.shape[0]
    else:
        vol_shape = label.shape[:-1]
        num_classes = label.shape[-1]
    num_subtypes = [mu[i].shape[0] for i in range(num_classes)]
    output_image = []
    for i in range(num_classes):
        subtype = np.random.multinomial(1, tau[i], size=vol_shape) # [*vol_shape, num_subtypes[i]]
        gaussians = np.stack([np.random.normal(mu[i][k], sigma[i][k], size=vol_shape) for k in range(num_subtypes[i])],
                             axis=-1)
        output_image.append(np.sum(subtype * gaussians, axis=-1))
    if channel_first:
        return np.sum(np.stack(output_image) * label, axis=0)
    else:
        return np.sum(np.stack(output_image, axis=-1) * label, axis=-1)


def get_ground_truth_parameters(images, labels, num_subtypes):
    """
    Compute the ground-truth parameters for mu and sigma2 using Torch

    :param images: image tensor of shape [1, num_subjects, 1, *vol_shape]
    :param labels: one-hot label tensor of shape [1, num_subjects, num_classes, *vol_shape]
    :param num_subtypes: a list of numbers for each class of labels.
    :return: mu - a list of arrays of shape [num_subjects, num_subtypes[i]];
             sigma2 - a list of arrays of shape [num_subjects, num_subtypes[i]]
    """
    num_classes = len(num_subtypes)
    mu = []
    sigma2 = []
    # print(images.shape, labels.shape)

    labels = torch.chunk(labels, chunks=num_classes, dim=2) # [1, num_subjects, 1,, *vol_shape]
    dimension = len(images.shape) - 3
    # print(dimension)

    for i in range(num_classes):
        mu_ = torch.sum(images * labels[i], dim=[0] + [k + 3 for k in range(dimension)]) / labels[i].sum(
            dim=[0] + [k + 3 for k in range(dimension)]) # [num_subjects, 1]
        # print(mu_.shape)
        # print(mu_.cpu().numpy())
        sigma2_ = torch.sum(
            (images - mu_.view(1, -1, 1, *[1] * dimension))**2 * labels[i], dim=[0] +
            [k + 3
             for k in range(dimension)]) / labels[i].sum(dim=[0] + [k + 3
                                                                    for k in range(dimension)]) # [num_subjects, 1]

        mu.append(mu_.repeat(1, num_subtypes[i]))

        sigma2.append(sigma2_.repeat(1, num_subtypes[i]) / num_subtypes[i])

    return mu, sigma2


def get_initial_parameters(images, prior, num_subtypes):
    """
    Compute the initial parameters for mu and sigma2 using Torch

    :param images: image tensor of shape [1, num_subjects, 1, *vol_shape]
    :param prior: one-hot label tensor of shape [1, num_classes, *vol_shape]
    :param num_subtypes: a list of numbers for each class of labels.
    :return: pi - a tensor of shape [num_classes]
             mu - a list of arrays of shape [num_subjects, num_subtypes[i]];
             sigma2 - a list of arrays of shape [num_subjects, num_subtypes[i]]
    """
    num_classes = len(num_subtypes)
    num_subjects = images.size(1)
    mu = []
    sigma2 = []
    # print(images.shape, labels.shape)

    priors = torch.chunk(prior, chunks=num_classes, dim=1) # [1, 1, *vol_shape]
    dimension = len(images.shape) - 3
    # print(dimension)
    pi = prior.sum(dim=(0, *[i + 2 for i in range(dimension)])) / prior.sum()

    for i in range(num_classes):
        mu_ = torch.sum(images * priors[i].unsqueeze(1), dim=[0] + [k + 3 for k in range(dimension)]) / priors[i].sum(
            dim=[0] + [k + 2 for k in range(dimension)]) # [num_subjects, 1]
        # print(mu_.shape)
        # print(mu_.cpu().numpy())
        sigma2_ = torch.sum(
            (images - mu_.view(1, -1, 1, *[1] * dimension))**2 * priors[i].unsqueeze(1), dim=[0] +
            [k + 3
             for k in range(dimension)]) / priors[i].sum(dim=[0] + [k + 2
                                                                    for k in range(dimension)]) # [num_subjects, 1]

        # print(sigma2_.cpu().numpy())
        if num_subtypes[i] == 1:
            mu.append(mu_.repeat(1, num_subtypes[i]))
        else:
            mu.append(mu_.repeat(1, num_subtypes[i]) + torch.rand([num_subjects, num_subtypes[i]]) * sigma2_.sqrt())

        sigma2.append(sigma2_.repeat(1, num_subtypes[i]) / num_subtypes[i])

    return pi, mu, sigma2


def get_one_hot_label(gt, label_intensities, channel_first=False):
    """
    Process label data into one-hot representation.

    :param gt: A ground-truth array, of shape [*vol_shape].
    :return: An array of one-hot representation, of shape [num_classes, *vol_shape].
    """
    num_classes = len(label_intensities)
    label = np.around(gt)
    if channel_first:
        label = np.zeros((np.hstack((num_classes, label.shape))), dtype=np.float32)

        for k in range(1, num_classes):
            label[k] = (gt == label_intensities[k])

        label[0] = np.logical_not(np.sum(label[1:, ], axis=0))
    else:
        label = np.zeros((np.hstack((label.shape, num_classes))), dtype=np.float32)

        for k in range(1, num_classes):
            label[..., k] = (gt == label_intensities[k])

        label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))

    return label


def visualize_image2d(image, **kwargs):
    """

    :param image: a image of shape [nx, ny, channel]
    :return:
    """
    plt.imshow(image, **kwargs)
    plt.show()


def _get_random_patch_center_covering_foreground(label, label_intensities, margin=(20, 20, 20)):
    """
    Crop random patches that cover the foreground.
    :param label: The label to crop patches.
    :param margin: The margin between the patch center and the foreground area.
    :return: A random patch center.
    """
    foreground_flag = np.any(
        np.concatenate(tuple([(np.expand_dims(label, axis=0) == k) for k in label_intensities[1:]])), axis=0)
    arg_index = np.argwhere(foreground_flag) # [n, 3]


def process_atlas(atlas):
    """
    Convert an atlas into a probabilistic one.

    :param atlas: of shape [nx, ny, nz]
    :return: The probabilistic atlas, of shape [1, nx, ny, nz, num_classes].
    """
    binary_atlas = get_one_hot_label(atlas, ) # [1, nx, ny, nz, num_classes]
    return atlas


def crop_to_shape(data, shape, ndim=None, **kwargs):
    """
    Crops the volumetric tensor or array into the given image shape by removing the border
    (expects a tensor or array of shape [*vol_shape, (channels)]).

    :param data: the tensor or array to crop, shape=[*vol_shape, (channels)]
    :param shape: the target shape
    :return: The cropped tensor or array.
    """
    if isinstance(shape, (tuple, list)):
        ndim = len(shape)
    elif isinstance(shape, (int, float)):
        assert ndim is not None
        shape = [shape] * ndim
    data_shape = np.asarray(data.shape[:ndim], dtype=np.int32)
    crop_shape = np.asarray(shape, dtype=np.int32)

    assert np.all(data_shape >= crop_shape), "The shape of array to be cropped is smaller than the target shape!"

    center = kwargs.pop('center', data_shape // 2)
    start = center - crop_shape // 2
    end = center + (crop_shape - crop_shape // 2)

    start = np.where(crop_shape == data_shape, 0, start)
    end = np.where(crop_shape == data_shape, data_shape, end)

    assert all(start >= 0) and all(end <= data_shape), "Start and End must be bounded! " \
                                                       "Start: %s, End: %s, Upper bound: %s" % (start, end, data_shape)

    for d in range(ndim):
        data = data.take(range(start[d], end[d]), d)

    return data


def crop_into_blocks(data, patch_size=None, block_size=None, num_blocks=(2, 2, 2), output_type='dict', **kwargs):
    """
    Crop the 3-D data into blocks, 

    :param data: The data array of shape [channels/n_class, *vol_shape, (n_atlas,) ].
    :param patch_size: The size of the patch located at the center of the data.
    :param num_blocks: The number of blocks along each spatial axis.
    :param block_size: The size of each block, can be overlapping.
    :param input_type: 'image' or 'label'.
    :param output_type: 'dict' or 'array'.
    :return: A dictionary containing blocks, with the key indicating the location of each block.
    """
    # assert input_type in ['image', 'label'], "The input type must be either 'image' or 'label'!"
    assert output_type in ['dict', 'array'], "The output type must be either 'dict' or 'array'!"

    if isinstance(num_blocks, int):
        num_blocks = (num_blocks, ) * 3

    assert num_blocks in [(1, 1, 1), (2, 2, 2)], "The number of blocks must be (1, 1, 1), (2, 2, 2) in this preliminary " \
                                               "version!"
    # Todo: Generalise the function into arbitrary number of blocks.
    data_size = np.asarray(data.shape[1:4], dtype=np.int16)

    if patch_size is None:
        patch_size = np.asarray(data.shape[1:4], dtype=np.int16)
    else:
        patch_size = np.asarray(patch_size, dtype=np.int16)

    if block_size is None:
        block_size = patch_size // np.asarray(num_blocks)
    else:
        block_size = np.asarray(block_size, dtype=np.int16)

    data_patch_gap_left = (data_size - patch_size) // 2
    data_patch_gap_right = data_patch_gap_left - (data_size - patch_size)

    if num_blocks == (1, 1, 1):
        return data[:,
                    data_patch_gap_left[0]:data_patch_gap_left[0] + patch_size[0],
                    data_patch_gap_left[1]:data_patch_gap_left[1] + patch_size[1],
                    data_patch_gap_left[2]:data_patch_gap_left[2] + patch_size[2],
                    ]

    data_block_gap = data_size - block_size

    codes = list(itertools.product(*[range(k) for k in num_blocks]))
    block_begins = [
        np.asarray(c, dtype=np.int16) * data_block_gap + data_patch_gap_left +
        (data_patch_gap_right - data_patch_gap_left) * c for c in codes
    ]
    block_ends = [(np.asarray(c, dtype=np.int16) - 1) * data_block_gap + data_size + data_patch_gap_left +
                  (data_patch_gap_right - data_patch_gap_left) * c for c in codes]
    block_indices = zip(block_begins, block_ends)

    blocks = []
    for begin, end in block_indices:
        blocks.append(data[:, begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], ])

    if output_type == 'dict':
        return dict(zip(codes, blocks))
    elif output_type == 'array':
        return np.concatenate(blocks, axis=0)


def sobel_filter(vol: torch.Tensor, return_norm: bool=False):
    """
    Compute edge magnitude from the Sobel filter

    Args:
        vol (torch.Tensor): tensor of shape [B, C, *vol_shape]
    """
    n_dim = vol.ndim - 2
    n_channel = vol.shape[1]
    
    diff_kernel = torch.as_tensor([1, 0, -1], dtype=vol.dtype, device=vol.device)
    weight_kernel = torch.as_tensor([1, 2, 1], dtype=vol.dtype, device=vol.device)
    
    if n_dim == 2:
        d_x = diff_kernel.view(1, 1, 1, 3).repeat(n_channel, 1, 1, 1)
        w_x = weight_kernel.view(1, 1, 3, 1).repeat(n_channel, 1, 1, 1)
        
        d_y = diff_kernel.view(1, 1, 3, 1).repeat(n_channel, 1, 1, 1)
        w_y = weight_kernel.view(1, 1, 1, 3).repeat(n_channel, 1, 1, 1)
        
        g_x = F.conv2d(F.conv2d(vol, d_x, 
                                padding='same', groups=n_channel), w_x, 
                        padding='same', groups=n_channel)
        g_y = F.conv2d(F.conv2d(vol, d_y,
                                padding='same', groups=n_channel), w_y, 
                        padding='same', groups=n_channel)
            
        if return_norm:
            g = torch.sqrt(g_x ** 2 + g_y ** 2)
        else:
            g = torch.stack([g_x, g_y], dim=-1)
                
    elif n_dim == 3:
        d_x = diff_kernel.view(1, 1, 1, 1, 3).repeat(n_channel, 1, 1, 1, 1)
        w1_x = weight_kernel.view(1, 1, 1, 3, 1).repeat(n_channel, 1, 1, 1, 1)
        w2_x = weight_kernel.view(1, 1, 3, 1, 1).repeat(n_channel, 1, 1, 1, 1)
        
        d_y = diff_kernel.view(1, 1, 1, 3, 1).repeat(n_channel, 1, 1, 1, 1)
        w1_y = weight_kernel.view(1, 1, 1, 1, 3).repeat(n_channel, 1, 1, 1, 1)
        w2_y = weight_kernel.view(1, 1, 3, 1, 1).repeat(n_channel, 1, 1, 1, 1)
        
        d_z = diff_kernel.view(1, 1, 3, 1, 1).repeat(n_channel, 1, 1, 1, 1)
        w1_z = weight_kernel.view(1, 1, 1, 3, 1).repeat(n_channel, 1, 1, 1, 1)
        w2_z = weight_kernel.view(1, 1, 1, 1, 3).repeat(n_channel, 1, 1, 1, 1)
        
        g_x = F.conv3d(F.conv3d(F.conv3d(vol, d_x, 
                                            padding='same', groups=n_channel), w1_x, 
                                padding='same', groups=n_channel), w2_x,
                        padding='same', groups=n_channel)
        g_y = F.conv3d(F.conv3d(F.conv3d(vol, d_y,
                                            padding='same', groups=n_channel), w1_y,
                                padding='same', groups=n_channel), w2_y, 
                        padding='same', groups=n_channel)
        g_z = F.conv3d(F.conv3d(F.conv3d(vol, d_z, 
                                            padding='same', groups=n_channel), w1_z, 
                                padding='same', groups=n_channel), w2_z, 
                        padding='same', groups=n_channel)
        
        if return_norm:
            g = torch.sqrt(g_x ** 2 + g_y ** 2 + g_z ** 2)
        else:
            g = torch.stack([g_x, g_y, g_z], dim=-1)
    else:
        raise NotImplementedError
    
    return g


def gauss_kernel1d(sigma):
    assert sigma >= 0
    if sigma == 0:
        return 1
    else:
        tail = math.floor(sigma*3)
        k = np.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / np.sum(k)


def ones_kernel1d(r):
    assert r >= 0
    if r == 0:
        return 1
    else:
        s = int(r)
        return np.ones([s])


def separable_filter3d(vol, kernel, mode='torch'):
    """
    3D convolution using separable filter along each axis

    :param vol: torch tensor of shape [batch, channels, *vol_shape]
    :param kernel: of shape [k]
    :return: of shape [batch, channels, *vol_shape]
    """
    if np.all(kernel == 0):
        return vol
    if mode == 'torch':
        kernel = torch.as_tensor(kernel, dtype=vol.dtype, device=vol.device)
        if vol.ndim == 3:
            vol = vol.unsqueeze(0).unsqueeze(0)
        channels = vol.size(1)
        kernel = kernel.repeat(channels, 1, 1, 1, 1)
        padding = kernel.size(-1) // 2
        return F.conv3d(
            F.conv3d(F.conv3d(vol, kernel.view(channels, 1, -1, 1, 1), padding=(padding, 0, 0), groups=channels),
                     kernel.view(channels, 1, 1, -1, 1), padding=(0, padding, 0), groups=channels),
            kernel.view(channels, 1, 1, 1, -1), padding=(0, 0, padding), groups=channels)
    elif mode == 'np':
        if vol.ndim == 2:
            vol = np.expand_dims(vol, axis=(0, 1))
        return signal.convolve(signal.convolve(signal.convolve(vol,
                                                               np.reshape(kernel, [1, 1, -1, 1, 1]), 'same'),
                                               np.reshape(kernel, [1, 1, 1, -1, 1]), 'same'),
                               np.reshape(kernel, [1, 1, 1, 1, -1]), 'same')


def separable_filter2d(vol, kernel, mode='torch'):
    """
    2D convolution using separable filter along each axis

    :param vol: torch tensor of shape [batch, channels, *vol_shape]
    :param kernel: of shape [k]
    :return: of shape [batch, channels, *vol_shape]
    """
    if np.all(kernel == 0):
        return vol
    if mode == 'torch':
        kernel = torch.as_tensor(kernel, dtype=vol.dtype, device=vol.device)
        if vol.ndim == 2:
            vol = vol.unsqueeze(0).unsqueeze(0)
        channels = vol.size(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        padding = kernel.size(-1) // 2
        return F.conv2d(F.conv2d(vol, kernel.view(channels, 1, -1, 1), padding=(padding, 0), groups=channels),
                        kernel.view(channels, 1, 1, -1), padding=(0, padding), groups=channels)
    elif mode == 'np':
        if vol.ndim == 2:
            vol = np.expand_dims(vol, axis=(0, 1))
        return signal.convolve(signal.convolve(vol, np.reshape(kernel, [1, 1, -1, 1]), 'same'),
                                               np.reshape(kernel, [1, 1, 1, -1]), 'same')


def circular_filter(vol, r, mode='torch'):
    """
    Convolve the volume with a circular filter of ones.

    :param vol: tensor of shape [B, C, *vol_shape]
    :param r: the radius
    :param mode:
    :return:
    """
    d = vol.ndim - 2
    n = math.ceil(2 * r) + 1
    kernel = np.ones([n] * d)
    grid = np.stack(np.meshgrid(*[range(n)] * d))  # [d, n, n]
    center = np.asarray([n / 2] * d).reshape([d, 1, 1])
    mask = np.square(grid - center).sum(0) <= r ** 2
    kernel *= mask
    if mode == 'torch':
        kernel = torch.as_tensor(kernel, device=vol.device, dtype=vol.dtype)
        channels = vol.size(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, *[1] * d)
        if d == 2:
            return F.conv2d(vol, kernel, padding='same', groups=channels)
        if d == 3:
            return F.conv3d(vol, kernel, padding='same', groups=channels)
    elif mode == 'np':
        return signal.convolve(vol, np.expand_dims(kernel, axis=[0, 1]), mode='same')
    else:
        raise NotImplementedError
