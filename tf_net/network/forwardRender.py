from poseVAE import PoseVAE
from imageGAN import ImageGAN
from numpy.matlib import repmat
from numpy.random import RandomState
import numpy as np
import time
import sys
import os
import cv2
sys.path.append('./')
import data.util
import globalConfig

class ForwardRender(object):
    def __init__(self, dim_x):
        self.dim_x = dim_x
        self.pose_vae = PoseVAE(dim_x=dim_x)
        self.alignment_layer = \
            self.build_latent_alignment_layer(self.pose_vae,
                                              self.origin_input_layer)
        # self.image_gan = ImageGAN(dim_z=selfdim_z)
        self.lr, self.b1 = 0.001, 0.5
        self.batch_size = 200
        print('vae and gan initialized')

        # print('all parameters: {}'.format(self.params))

    def build_latent_alignment_layer(self, pose_vae,
                                     origin_layer=None,
                                     quad_layer=None):
        self.pose_z_dim=pose_vae.dim_z
