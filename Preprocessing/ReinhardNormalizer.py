#!/usr/bin/env python3
#-*- coding: utf-8
# From: https://github.com/SBU-BMI/quip_cnn_segmentation/tree/develop/large_scale_training_testing

from __future__ import division

import os
import numpy as np
import skimage
from skimage import color
from .PImage import PImage

### Main class ###

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def _standardize_brightness(self,I):
        """

        :param I:
        :return:
        """
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

    def _get_mean_std(self,I):
        """
        Get mean and standard deviation of each channel
        :param I: uint8
        :return:
        """
        I1, I2, I3 = self._lab_split(I)
        m1 = np.mean(I1)
        sd1 = np.std(I1)
        m2 = np.mean(I2)
        sd2 = np.std(I2)
        m3 = np.mean(I3)
        sd3 = np.std(I3)

        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds

    def _merge_back(self,I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8
        :param I1:
        :param I2:
        :param I3:
        :return:
        """
        I1 *= 2.55
        I2 += 128.0
        I3 += 128.0
        I = np.clip(np.dstack((I1, I2, I3)), 0, 255)
        I = color.lab2rgb(I)
        I = skimage.img_as_ubyte(I)
        return I

    def _lab_split(self,I):
        """
        Convert from RGB uint8 to LAB and split into channels
        :param I: uint8
        :return:
        """
        I = color.rgb2lab(I)
        I = I.astype(np.float32)
        I1, I2, I3 = I[:,:,0],I[:,:,1],I[:,:,2]
        I1 /= 2.55
        I2 -= 128.0
        I3 -= 128.0
        #print("Dimension shapes: {}, {}, {}".format(I1.shape,I2.shape,I3.shape))
        return I1, I2, I3

    def fit(self, target):
        target = self._standardize_brightness(target)
        means, stds = self._get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        I = self._standardize_brightness(I)
        I1, I2, I3 = self._lab_split(I)
        means, stds = self._get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / (1e-10+stds[0]))) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / (1e-10+stds[1]))) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / (1e-10+stds[2]))) + self.target_means[2]
        return self._merge_back(norm1, norm2, norm3)

class ReinhardNormalizer(object):
    def __init__(self, target_file):
        """
        target_file <str>: path to an image that should be used as the reference for normalization
        """
        if isinstance(target_file,str) and os.path.isfile(target_file):
            target_40X = PImage(target_file).readImage(keepImg=False,toFloat=False)
        else:
            raise ValueError("[ReinhardNormalizer] Target file should be an image file")
        
        self.n_40X = Normalizer()
        self.n_40X.fit(target_40X)

    def normalize(self, image):
        # image RGB in uint8
        return self.n_40X.transform(image)
