#!/usr/bin/env python3
#-*- coding: utf-8

__all__ = ['VGG16','UNet','Inception','SmallNet','Xception','ResNet50','ResNet101']

from .VGG import VGG16,EFVGG16
from .VGG import BayesVGG16, BayesVGG16A2
from .KMNIST import KNet,BayesKNet,GalKNet
from .EKNet import BayesEKNet
from .InceptionV4 import Inception,EFInception
from .ALTransf import SmallNet
from .Xception import Xception
from .ResNet import ResNet50,ResNet101,ResNet50V2
