from numpy.matlib import repmat
from numpy.random import RandomState
import numpy as np
import time
import sys
import os
import cv2
sys.path.append('./')
import globalConfig
import data.util
from depthGAN import DepthGAN
from poseVAE import PoseVAE
class ForwardRender(object):
    def __init__(self, dim_x):
        self.dim_x = dim_x
        self.pose_vae = PoseVAE(x_dim=x_dim)
