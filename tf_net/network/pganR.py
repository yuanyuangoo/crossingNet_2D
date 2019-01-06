import tensorflow as tf
from tqdm import tqdm

# from poseVAE import PoseVAE
# from imageGAN import ImageGAN
from p2Igan import p2igan
from p2pgan import P2PGAN
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
EPS = 1e-12
gray2rgb = tf.image.grayscale_to_rgb


class PganR(object):
    def __init__(self, dim_x=17*3, label_dim=15, sample_dir="samples", checkpoint_dir="./checkpoint"):
        self.checkpoint_dir = os.path.join(
            globalConfig.pganR_pretrain_path, checkpoint_dir)
        self.sample_dir = os.path.join(
            globalConfig.pganR_pretrain_path, sample_dir)
        self.dim_x = dim_x

        self.FR = p2igan(dim_z=self.dim_x, label_dim=label_dim)
        self.sample_G = self.FR.sample
        self.FR_G = self.FR.G
        self.p2p = P2PGAN(label_dim=label_dim)

        with tf.variable_scope("p2p") as scope:
            scope.reuse_variables()
            self.sample = self.p2p.build_generator(
                tf.image.grayscale_to_rgb(binary_activation(self.sample_G,0.4)), 3, reuse=True, is_training=False)

            self.G = self.p2p.build_generator(tf.image.grayscale_to_rgb(
                self.FR_G), 3, reuse=True, is_training=True)
            self.D = self.p2p.build_discriminator(
                gray2rgb(self.p2p.image_input), self.p2p.image_target, reuse=True)
            self.D_ = self.p2p.build_discriminator(
                gray2rgb(self.p2p.image_input), self.G, reuse=True)

            self.d_sum = histogram_summary("d", self.D)
            self.d__sum = histogram_summary("d_", self.D_)
            self.G_sum = image_summary("G", self.G)

            self.g_loss_GAN = tf.reduce_mean(-tf.log(self.D_ + EPS))
            mask = ((-1*self.p2p.image_input)+1)/2
            self.g_loss_L1 = tf.reduce_mean(
                tf.abs(tf.multiply(self.p2p.image_target - self.G, mask)))
            self.g_loss = self.g_loss_GAN * self.p2p.gan_weight + \
                self.p2p.g_loss_L1 * self.p2p.l1_weight
            self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

            self.d_loss = tf.reduce_mean(-(tf.log(self.D + EPS) +
                                           tf.log(1 - self.D_ + EPS)))
            self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

            self.g_var = self.p2p.g_vars
            self.d_var=self.p2p.d_vars
            self.saver = tf.train.Saver()

        self.batch_size = self.FR.batch_size
        
        # self.d_loss=self.p2p.d_loss


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

            d_optim = tf.train.AdamOptimizer(self.p2p.learning_rate, beta1=self.p2p.beta1) \
                .minimize(self.d_loss, var_list=self.d_var)
            g_optim = tf.train.AdamOptimizer(self.p2p.learning_rate, beta1=self.p2p.beta1) \
                .minimize(self.g_loss, var_list=self.g_var)

            # g_optim_fr = tf.train.AdamOptimizer(self.p2p.learning_rate, beta1=self.p2p.beta1) \
            #     .minimize(self.g_loss_fr, var_list=forward_gan_var)

            rest_var = [
                val for val in tf.global_variables() if val not in forward_gan_var and val not in p2p_gan_var]
            
            init_new_vars_op = tf.variables_initializer(rest_var).run()


            for epoch in range(self.epoch):
                for idx in range(0, int(batch_idxs)):
                    batch_target = train_img_rgb[idx *
                                                 self.batch_size:(idx+1)*self.batch_size]
                    batch_img = train_img[idx *
                                          self.batch_size:(idx+1)*self.batch_size]
                    batch_labels = train_labels[idx *
                                                self.batch_size:(idx+1)*self.batch_size]
                    batch_skel = train_skel[idx *
                                            self.batch_size:(idx+1)*self.batch_size]
                    # noise = noisy('s&p', batch_target)

                    _, _,  d_loss, g_loss = self.sess.run([d_optim, g_optim, self.d_loss, self.g_loss], feed_dict={
                        self.FR.pose_input: batch_skel,
                        self.FR.y: batch_labels,
                        self.p2p.label: batch_labels,
                        self.p2p.image_target: batch_target,
                        self.p2p.image_input: batch_img,
                        # self.p2p.image_input_n: noise,
                        self.FR.image_target: batch_img
                    })
                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, d_loss, g_loss))

                    if np.mod(counter, 200) == 1:
                        # show_all_variables()
                        sample_G, samples, _ = self.sess.run([self.sample_G, self.sample, self.p2p.image_target], feed_dict={
                            self.FR.pose_input: test_skel,
                            self.FR.y: test_labels,
                            self.p2p.label: test_labels,
                            self.p2p.image_target: test_img_rgb
                        })
                        save_images(sample_G, image_manifold_size(sample_G.shape[0]),
                                    '{}/G_train_{:04d}.png'.format(self.sample_dir, counter), skel=test_skel)
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}/train_{:04d}.png'.format(self.sample_dir, counter), skel=test_skel)
                        # save_images(real_image, image_manifold_size(real_image.shape[0]),
                        #             '{}/real_train_{:04d}.png'.format(self.sample_dir, counter), skel=test_skel)
                    if np.mod(counter, 5000) == 1:
                        self.save(self.checkpoint_dir, counter)

            self.save(self.checkpoint_dir, counter)

    def predict(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        test_skel = np.load('samples_skel.out.npy')
        test_label = np.load('samples_label.out.npy')
        print("Samples loaded...")
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
            batch_idxs = test_skel.shape[0]//self.batch_size
            for idx in tqdm(range(0, int(batch_idxs))):
                batch_labels = test_label[idx *
                                          self.batch_size:(idx+1)*self.batch_size]
                batch_skel = test_skel[idx *
                                       self.batch_size:(idx+1)*self.batch_size]
                samples = self.sess.run(self.sample, feed_dict={
                    self.FR.pose_input: batch_skel,
                    self.FR.y: batch_labels,
                    self.p2p.label: batch_labels
                })
                # save_images(sample_G, image_manifold_size(sample_G.shape[0]),
                #             '{}/test_G_{:04d}.png'.format(self.sample_dir, idx), skel=batch_skel)
                # save_images(samples, image_manifold_size(samples.shape[0]),
                #             '{}/test_S{:04d}.png'.format(self.sample_dir, idx), skel=batch_skel)
                # save_images(samples, image_manifold_size(samples.shape[0]),
                #             '{}/test_{:04d}.png'.format(self.sample_dir, idx), skel=None)
                save_images_one_by_one(
                    samples, self.sample_dir+'_predicted', idx)
    def test(self, valid_dataset):
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

            sample_G,samples,real_image = self.sess.run([self.sample_G,self.sample,self.p2p.image_target], feed_dict={
                self.FR.pose_input: test_skel,
                self.FR.y: test_labels,
                self.p2p.label: test_labels,
                self.p2p.image_target: test_img_rgb
            })

            save_images(sample_G, image_manifold_size(sample_G.shape[0]),
                        '{}/test_G.png'.format(self.sample_dir), skel=test_skel)
            save_images(samples, image_manifold_size(samples.shape[0]),
                        '{}/test.png'.format(self.sample_dir), skel=test_skel)
            save_images(samples, image_manifold_size(samples.shape[0]),
                        '{}/test.png'.format(self.sample_dir))
            save_images(real_image, image_manifold_size(real_image.shape[0]),
                        '{}/test_real.png'.format(self.sample_dir), skel=test_skel)

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
        # import data.h36m as h36m
        # ds = Dataset()
        # ds.loadH36M(40960, mode='train', tApp=True, replace=False)

        # val_ds = Dataset()
        # val_ds.loadH36M(64, mode='valid', tApp=True, replace=True)
        pganR = PganR(dim_x=17*3)
    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(64*300, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadApe(64, mode='valid', tApp=True, replace=False)

        pganR = PganR(dim_x=15*3)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    # pganR.test(val_ds)
    # pganR.train(ds, val_ds)
    pganR.predict()
