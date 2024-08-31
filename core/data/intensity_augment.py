from skimage.exposure import rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
import numpy as np
import random


def randomIntensityFilter(im, dtype=np.float32):
    """
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaling and applies it on the input image.
    filters include: equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    """
    # Filters = [identity, equalizeHist, randomLog, randomSigmoid, randomGamma, randomGaussian, randomIntensity, randomNoise]
    Filters = [identity, randomLog, randomSigmoid, randomGamma, randomGaussian, randomIntensity, randomNoise]
    filter = random.choice(Filters)
    return np.asarray(filter(im), dtype=dtype)


def identity(x):
    return x


def randomLog(im):
    im = im.astype(np.int16)
    return adjust_log(im, gain=randRange(0.8, 1.2))


def randomSigmoid(im):
    im = im.astype(np.int16)
    return adjust_sigmoid(im, cutoff=randRange(0.4, 0.6), gain=randRange(8, 12))


def randRange(a, b):
    """
    a utility function to generate random float values in desired range
    """
    return np.random.rand() * (b - a) + a


def randomIntensity(im):
    """
    rescales the intensity of the image to random interval of image intensity distribution
    """
    im = im.astype(np.int16)
    return rescale_intensity(im,
                             in_range=tuple(np.percentile(im, (randRange(0, 5), randRange(95, 100)))),
                             out_range=tuple(np.percentile(im, (randRange(0, 5), randRange(95, 100)))))


def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    im = im.astype(np.int16)
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))


def randomGaussian(im):
    '''
    Gaussian filter for blurring the image with random variance.
    '''
    im = im.astype(np.float32)
    return gaussian(im, sigma=randRange(0, 2), channel_axis=None, preserve_range=True)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    rand_im = np.random.normal(loc=im, scale=np.ptp(im) / 100)
    return rand_im