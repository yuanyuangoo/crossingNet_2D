from imageGAN import batch_norm, conv2d, deconv2d
import tensorflow as tf
import globalConfig
import data.util
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


class ForwardRender(object):
    def __init__(self, dim_x):
        self.dim_x = dim_x
        self.pose_vae = PoseVAE(dim_x=dim_x)
        self.origin_input = tf.placeholder(tf.float32, shape=(None, 3))
        self.dim_z = 100
        self.alignment = \
            self.build_latent_alignment_layer(self.pose_vae,
                                              self.origin_input)


        self.image_gan = ImageGAN()
        self.render=self.image_gan.G

        self.lr, self.b1 = 0.001, 0.5
        self.batch_size = 200
        print('vae and gan initialized')

        # print('all parameters: {}'.format(self.params))

    def build_latent_alignment_layer(self, pose_vae,
                                     origin_layer=None,
                                     quad_layer=None):
        self.pose_z_dim = pose_vae.z.shape[1]
        self.z_dim = self.pose_z_dim
        if origin_layer is not None:
            self.z_dim += 3
        if quad_layer is not None:
            self.z_dim += 4build_latent_alignment_layer

        latent = pose_vae.z
        if origin_layer is not None:
            latent = tf.concat([latent, origin_layer], axis=1)
        if quad_layer is not None:
            latent = tf.concat([latent, quad_layer], axis=1)

        print('latent output shape = {}'
              .format(latent.shape))
        self.latent = latent

        latent = tf.placeholder(tf.float32, shape=(None, self.z_dim))
        self.bn1 = batch_norm(name="ali_bn1")
        alignment = self.bn1(tf.layers.dense(latent, self.z_dim, use_bias=True,name='ali_conv'))
        
        return alignment


ForwardRender(100)
