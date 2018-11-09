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
from imageGAN import ImageGAN
from poseVAE import PoseVAE
class ForwardRender(object):
    def __init__(self, dim_x):
        self.dim_x = dim_x
        self.pose_vae = PoseVAE(dim_x=dim_x)
        self.image_gan=ImageGAN(dim_z=self.dim_z)
        self.lr, self.b1 = 0.001, 0.5
        self.batch_size = 200
        print ('vae and gan initialized')
        self.params = self.pose_vae.encoder_params +\
                self.alignment_params +\
                self.depth_gan.gen_params
        print ('all parameters: {}'.format(self.params))