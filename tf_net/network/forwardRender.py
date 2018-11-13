from imageGAN import batch_norm, conv2d, deconv2d
import tensorflow as tf
from tqdm import tqdm
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
import globalConfig
from data.util import show_all_variables

class ForwardRender(object):
    def __init__(self, dim_x):
        self.keep_prob=1.0
        self.dim_x = dim_x
        self.pose_vae = PoseVAE(dim_x=dim_x)
        self.origin_input = tf.placeholder(tf.float32, shape=(None, 3))
        self.dim_z = 100
        self.alignment = \
            self.build_latent_alignment_layer(self.pose_vae,
                                              self.origin_input)
        self.x_hat = tf.placeholder(
            tf.float32, shape=[None, dim_x], name='input_pose')
        self.x = tf.placeholder(
            tf.float32, shape=[None, dim_x], name='target_pose')
        self.image_gan = ImageGAN()
        self.render = self.image_gan.G

        self.lr, self.b1 = 0.001, 0.5
        self.batch_size = 200
        print('vae and gan initialized')
        show_all_variables()
        # print('all parameters: {}'.format(self.params))
        self.real_image_var = tf.placeholder(
            name='real_image', dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(
            self.render, self.real_image_var)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build_latent_alignment_layer(self, pose_vae,
                                     origin_layer=None,
                                     quad_layer=None):
        self.pose_z_dim = pose_vae.z.shape[1]
        self.z_dim = self.pose_z_dim
        if origin_layer is not None:
            self.z_dim += 3
        if quad_layer is not None:
            self.z_dim += 4

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
        alignment = self.bn1(tf.layers.dense(
            latent, self.z_dim, use_bias=True, name='ali_conv'))
        
        return alignment

    def train(self, nepoch,  train_dataset, valid_dataset, desc='dummy'):
        cache_dir = os.path.join(
            globalConfig.model_dir, 'render/%s_%s' % (globalConfig.dataset, desc))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        img_dir = os.path.join(cache_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        train_size = len(train_dataset.frmList)
        self.x_hat = tf.placeholder(
            tf.float32, shape=[None, self.dim_x], name='input_pose')
        train_data = []
        train_labels = []
        for frm in train_dataset.frmList:
            train_data.append(frm.skel)
            train_labels.append(frm.label)

        test_data = []
        test_labels = []
        for frm in valid_dataset.frmList:
            test_data.append(frm.skel)
            test_labels.append(frm.label)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        train_data = train_data/max(-1*train_data.min(), train_data.max())
        test_data = test_data/max(-1*test_data.min(), test_data.max())
        VALIDATION_SIZE = 5000  # Size of the validation set.

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, :]
        validation_labels = train_labels[:VALIDATION_SIZE, :]

        train_data = train_data[VALIDATION_SIZE:, :]
        train_labels = train_labels[VALIDATION_SIZE:, :]

        valid_data = np.concatenate(
            (validation_data, validation_labels), axis=1)
        test_data = np.concatenate(
            (test_data, test_labels), axis=1)


        train_total_data = np.concatenate(
            (train_data, train_labels), axis=1)
        NUM_LABELS=15
        train_data_ = train_total_data[:, :-NUM_LABELS]
        train_size = train_total_data.shape[0]
        print('[ForwardRender] enter training loop with %d epoches' % nepoch)
        seed = 42
        np_rng = RandomState(seed)
        train_size = train_total_data.shape[0]
        n_samples = train_size
        total_batch = int(n_samples / self.batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(),
                     feed_dict={self.keep_prob: 0.9})
            for epoch in tqdm(range(nepoch)):
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * self.batch_size) % (n_samples)

                    batch_xs_input = train_data_[
                        offset:(offset + self.batch_size), :]
                    batch_xs_target = batch_xs_input

                    _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                        (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 0.9})
                    if epoch % 10 == 0:
                        batch_xs_input = test_data[
                            offset:(offset + self.batch_size), :]
                        batch_xs_target = batch_xs_input
                        for idx in range(0, len(batch_xs_input), 10):
                            img = sess.run(
                                self.render, feed_dict={self.x_hat: batch_xs_input, self.keep_prob: 1})
                            cv2.imwrite(os.path.join(
                                img_dir, '%d_%d.jpg' % (epoch, idx)), img)

    def resumePose(self, norm_pose, tran, quad=None):
        orig_pose = norm_pose.copy()
        orig_pose.shape = (-1,3)
        if quad is not None:
            R = np.matrix(quad)
            orig_pose = np.dot(R.transpose(), orig_pose.transpose())
        translation = repmat(tran.reshape((1,3)), orig_pose.shape[0], 1)
        orig_pose = translation + orig_pose
        orig_pose = orig_pose.flatten()
        return orig_pose

    def visPair(self, image, pose=None, trans=None, com=None, ratio=None):
        img = image[0].copy()
        img = (img+1)*127.0
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if pose is None:
            return img
        
        skel = pose.copy()
        skel.shape = (-1, 3)
        skel = skel*ratio
        skel2 = []
        for pt in skel:
            # pt2 = Camera.to2D(pt+com)
            pt2[2] = 1.0
            pt2 = np.dot(trans, pt2)
            pt2.shape = (3,1)
            pt2 = (pt2[0],pt2[1])
            skel2.append(pt2)
        for idx, pt2 in enumerate(skel2):
            cv2.circle(img, pt2, 3, 
                       data.util.figColor[colorPlatte[idx]], -1)
        for b in bones:
            pt1 = skel2[b[0]]
            pt2 = skel2[b[1]]
            color = b[2]
            cv2.line(img,pt1,pt2,color,2)
        return img