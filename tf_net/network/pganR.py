import tensorflow as tf
from tqdm import tqdm

from poseVAE import PoseVAE
from imageGAN import ImageGAN
from forwardRender import ForwardRender
from p2pgan import P2PGAN
from numpy.matlib import repmat
from numpy.random import RandomState
import numpy as np
import time
import sys
import os
import cv2
sys.path.append('./')
import globalConfig
from data.layers import *
from data.dataset import *
from data.util import *
from data.ops import *

Num_of_Joints = 17


class PganR(object):
    def __init__(self, dim_x=Num_of_Joints*3,sample_dir="samples",checkpoint_dir="./checkpoint"):
        self.checkpoint_dir = os.path.join(
            globalConfig.Forward_Render_pretrain_path, checkpoint_dir)
        self.sample_dir = os.path.join(
            globalConfig.Forward_Render_pretrain_path, sample_dir)
        self.dim_x = dim_x


        
        self.FR = ForwardRender(self.dim_x)
        self.p2p = P2PGAN(mode='test')
        with tf.variable_scope("p2p") as scope:
            scope.reuse_variables()
            self.sample = self.p2p.build_generator(
                tf.image.grayscale_to_rgb(self.FR.sample), 3, reuse=True, is_training=False)

        self.image_input_sum = histogram_summary(
            "image_input", self.p2p.image_input)
        self.batch_size = self.p2p.batch_size
        self.p2p.label = self.FR.pose_vae.label_hat

    def test(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        _, _, _, test_labels, test_skel, test_img, _, _ = prep_data(
            train_dataset, valid_dataset, self.batch_size)
        with tf.Session() as self.sess:
            forward_gan_var = [val for val in tf.trainable_variables(
            ) if 'p2p' not in val.name]
            self.saver = tf.train.Saver(forward_gan_var)
            _, _ = self.load(
                self.FR.checkpoint_dir)

            p2p_gan_var = [val for val in tf.trainable_variables(
            ) if 'p2p' in val.name]
            self.saver = tf.train.Saver(p2p_gan_var)
            could_load, checkpoint_counter = self.load(
                self.p2p.checkpoint_dir)
            
            nsamples = test_img.shape[0]
            counter = 1
            start_time = time.time()
            batch_idxs = int(nsamples/self.batch_size)
            for idx in tqdm(xrange(0, int(batch_idxs))):
                batch_labels = test_labels[idx *
                                           self.batch_size:(idx+1)*self.batch_size]
                batch_skels = test_skel[idx *
                                        self.batch_size:(idx+1)*self.batch_size]
                batch_img = test_img[idx *
                                     self.batch_size:(idx+1)*self.batch_size]

                samples = self.sess.run([self.sample], feed_dict={
                    self.FR.pose_input: batch_skels,
                    self.FR.label: batch_labels,
                    self.FR.image_gan.y: batch_labels
                })
                save_images(samples, image_manifold_size(samples.shape[0]),
                            '{}/test_{:04d}.png'.format(self.sample_dir, idx))
                counter += 1

    @property
    def model_dir(self):
        return "{}_{}".format(
            globalConfig.dataset, self.batch_size,
        )

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(
                self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        model_name = "Forward_Render.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        # for i in range(0, 20000, 20000):
        ds.loadH36M(10240, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        # for i in range(0, 20000, 20000):
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    pganR = PganR()
    pganR.test(ds, val_ds)
