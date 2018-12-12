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
import pprint as pp
Num_of_Joints = 17


class PganR(object):
    def __init__(self, dim_x=Num_of_Joints*3,sample_dir="samples",checkpoint_dir="./checkpoint"):
        self.checkpoint_dir = os.path.join(
            globalConfig.pganR_pretrain_path, checkpoint_dir)
        self.sample_dir = os.path.join(
            globalConfig.pganR_pretrain_path, sample_dir)
        self.dim_x = dim_x


        
        self.FR = ForwardRender(self.dim_x)
        self.p2p = P2PGAN(mode='test')
        self.sample_G = self.FR.sample


        with tf.variable_scope("p2p") as scope:
            scope.reuse_variables()
            self.sample = self.p2p.build_generator(
                tf.image.grayscale_to_rgb(self.sample_G), 3, reuse=True, is_training=False)

        self.batch_size = self.FR.batch_size
        self.p2p.label = self.FR.pose_vae.label_hat
        self.g_loss = self.p2p.g_loss
        self.d_loss = self.p2p.d_loss



    def train(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        train_labels, train_skel, train_img, train_img_rgb, _, batch_idxs = prep_data(
            train_dataset, self.batch_size)
        test_labels, test_skel, test_img, test_img_rgb, _, _ = prep_data(
            valid_dataset, self.batch_size)
        with tf.Session() as self.sess:
            d_optim = tf.train.AdamOptimizer(self.p2p.learning_rate, beta1=self.p2p.beta1) \
                .minimize(self.d_loss)
            g_optim = tf.train.AdamOptimizer(self.p2p.learning_rate, beta1=self.p2p.beta1) \
                .minimize(self.g_loss)
            # self.g_sum = merge_summary([self.image_input_sum, self.d__sum,
            #                             self.G_sum, self.g_loss_sum])
            # self.d_sum = merge_summary(
            #     [self.image_input_sum, self.image_target_sum, self.d_sum,  self.d_loss_sum])
            self.writer = SummaryWriter(
                os.path.join(globalConfig.p2p_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.pganR')

            tf.global_variables_initializer().run()
            forward_gan_var = [
                val for val in tf.global_variables() if 'p2p' not in val.name]
            self.saver = tf.train.Saver(forward_gan_var)
            _, _ = self.load(self.FR.checkpoint_dir)

            p2p_gan_var = [val for val in tf.global_variables(
            ) if 'p2p' in val.name]
            self.saver = tf.train.Saver(p2p_gan_var)
            could_load, checkpoint_counter = self.load(
                self.p2p.checkpoint_dir)

            nsamples = train_img.shape[0]
            counter = 1
            start_time = time.time()
            self.epoch=200

            for epoch in xrange(self.epoch):
                for idx in xrange(0, int(batch_idxs)):
                    batch_target = train_img_rgb[idx *
                                                 self.batch_size:(idx+1)*self.batch_size]
                    batch_labels = train_labels[idx *
                                                self.batch_size:(idx+1)*self.batch_size]
                    batch_skel = train_skel[idx *
                                            self.batch_size:(idx+1)*self.batch_size]

                    _, _, d_loss, g_loss = self.sess.run([d_optim, g_optim, self.d_loss, self.g_loss], feed_dict={
                        self.FR.pose_input: batch_skel,
                        self.FR.pose_vae.label_hat: batch_labels,
                        self.FR.image_gan.y: batch_labels,
                        self.p2p.label: batch_labels,
                        self.p2p.image_target: batch_target
                    })
                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, d_loss, g_loss))

                    if np.mod(counter, 100) == 1:
                        # show_all_variables()
                        sample_G,samples = self.sess.run([self.sample_G,self.sample], feed_dict={
                            self.FR.pose_input: test_skel,
                            self.FR.pose_vae.label_hat: test_labels,
                            self.FR.image_gan.y: test_labels,
                            self.p2p.label: test_labels
                        })
                    save_images(sample_G, image_manifold_size(sample_G.shape[0]),
                                '{}/train_{:04d}_G.png'.format(self.sample_dir, idx))
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                '{}/train_{:04d}.png'.format(self.sample_dir, idx))

                    if np.mod(counter, 500) == 1:
                        self.save(self.checkpoint_dir, counter)

    def test(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        test_labels, test_skel, test_img, test_img_rgb, _, _ = prep_data(
            valid_dataset, self.batch_size)
        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()

            forward_gan_var = [
                val for val in tf.global_variables() if 'p2p' not in val.name]
            self.saver = tf.train.Saver(forward_gan_var)
            _, _ = self.load(self.FR.checkpoint_dir)

            p2p_gan_var = [val for val in tf.global_variables(
            ) if 'p2p' in val.name]
            self.saver = tf.train.Saver(p2p_gan_var)
            could_load, checkpoint_counter = self.load(
                self.p2p.checkpoint_dir)
            
            nsamples = test_img.shape[0]
            counter = 1
            start_time = time.time()

            sample_G,samples = self.sess.run([self.sample_G,self.sample], feed_dict={
                self.FR.pose_input: test_skel,
                self.FR.pose_vae.label_hat: test_labels,
                self.FR.image_gan.y: test_labels,
                self.p2p.label: test_labels
            })
            idx = 0

            save_images(sample_G, image_manifold_size(sample_G.shape[0]),
                        '{}/test_{:04d}_G.png'.format(self.sample_dir, idx))
            save_images(samples, image_manifold_size(samples.shape[0]),
                        '{}/test_{:04d}.png'.format(self.sample_dir, idx))
            counter += 1

    @property
    def model_dir(self):
        return "{}_{}".format(
            globalConfig.dataset, self.batch_size)

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
            print(" [*] Failed to find a ch10240kpoint")
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
    # pganR.test(ds, val_ds)
    pganR.train(ds, val_ds)
