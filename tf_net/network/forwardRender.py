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
from data.layers import *

class ForwardRender(object):
    def __init__(self, dim_x):
        self.keep_prob = 1.0
        self.dim_x = dim_x
        self.pose_vae = PoseVAE(dim_x=dim_x)
        self.origin_input = tf.placeholder(tf.float32, shape=(None, 3),name="origin")
        self.origin_input_sum = histogram_summary("origin_input", self.origin_input)

        self.image_gan = ImageGAN()
        self.dim_z = self.image_gan.dim_z

        self.render = self.build_latent_alignment_layer(self.pose_vae)
        _, self.dis_px_layer, self.feamat_layer,  = self.image_gan.build_discriminator(
            self.render, self.image_gan.y, reuse=True)
        self.render_sum = image_summary("render", self.render)

        self.lr, self.b1 = 0.001, 0.5
        self.batch_size = self.pose_vae.batch_size
        print('vae and gan initialized')
        # show_all_variables()
        # print('all parameters: {}'.format(self.params))

        self.pose_input = self.pose_vae.x_hat

        #Train Ali
        self.real_image = self.image_gan.inputs
        self.pixel_loss = tf.losses.mean_squared_error(
            self.real_image, self.render)
        t_vars = tf.trainable_variables()
        self.alignment_vars = [var for var in t_vars if 'ali' in var.name]

        self.forwarRender_vars = self.pose_vae.encoder_vars + \
            self.alignment_vars+self.image_gan.g_vars

        self.ali_train_op = tf.train.AdamOptimizer(
            self.lr, self.b1).minimize(self.pixel_loss, var_list=self.forwarRender_vars)



    def build_latent_alignment_layer(self, pose_vae,
                                     origin_layer=None,
                                     quad_layer=None):
        
        self.pose_z_dim = int(pose_vae.z.shape[1])
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
        with tf.variable_scope("ali", reuse=False) as scope:
            print('latent output shape = {}'
                  .format(latent.shape))
            self.latent = latent

            # use None input, to adapt z from both pose-vae and real-test
            self.bn = batch_norm(name="bn1")
            self.alignment = self.bn(
                tf.contrib.layers.fully_connected(self.latent, num_outputs=self.image_gan.dim_z, activation_fn=None))
        render = self.image_gan.build_sampler(self.alignment, self.image_gan.y)
        return render

    def train(self, nepoch,  train_dataset, valid_dataset, desc='dummy'):
        cache_dir = os.path.join(
            globalConfig.model_dir, 'render/%s_%s' % (globalConfig.dataset, desc))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        img_dir = os.path.join(cache_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        train_size = len(train_dataset.frmList)

        train_skel = []
        train_labels = []
        train_img=[]
        for frm in train_dataset.frmList:
            train_skel.append(frm.skel)
            train_labels.append(frm.label)
            train_img.append(frm.norm_img)

        test_skel = []
        test_labels = []
        test_img=[]
        for frm in valid_dataset.frmList:
            test_skel.append(frm.skel)
            test_labels.append(frm.label)
            test_img.append(frm.norm_img)

        train_skel = np.asarray(train_skel)
        train_labels = np.asarray(train_labels)
        train_img = np.asarray(train_skel)

        test_skel = np.asarray(test_skel)
        test_labels = np.asarray(test_labels)
        test_img=np.asarray(test_img)

        train_skel = train_skel/max(-1*train_skel.min(), train_skel.max())
        test_skel = test_skel/max(-1*test_skel.min(), test_skel.max())

        train_size = train_skel.shape[0]
        
        print('[ForwardRender] enter training loop with %d epoches' % nepoch)
        total_batch = int(train_size / self.batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(),
                     feed_dict={self.keep_prob: 0.9})
            for epoch in tqdm(range(nepoch)):
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * self.batch_size) % (total_batch)
                    _, tot_loss = sess.run((self.ali_train_op, self.pixel_loss), feed_dict={
                        self.real_image: train_img[offset:(offset + self.batch_size), :],
                        self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                        self.origin_input: None,
                        self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                        self.pose_vae.keep_prob: 0.9
                    })
                    print("epoch %d: L_tot %03.2f" % (epoch, tot_loss))

                if epoch % 10 == 0:
                    for idx in range(0, self.batch_size, 10):
                        self.render.eval({
                            self.pose_input: test_skel[offset+idx, :, :],
                            self.origin_input: None,
                            self.image_gan.y: test_labels[offset+idx, :],
                            self.pose_vae.keep_prob: 1
                        })
                        cv2.imwrite(os.path.join(
                            img_dir, '%d_%d.jpg' % (epoch, idx)), self.render)

    def resumePose(self, norm_pose, tran, quad=None):
        orig_pose = norm_pose.copy()
        orig_pose.shape = (3, -1)
        if quad is not None:
            R = np.matrix(quad)
            orig_pose = np.dot(R.transpose(), orig_pose.transpose())
        if tran.shape[0].value is not None:
            translation = repmat(tran.reshape((1, 3)), orig_pose.shape[0], 1)
            orig_pose = translation + orig_pose
        orig_pose = orig_pose.flatten()
        return orig_pose

    def visPair(self, image, pose=None, trans=None, com=None, ratio=None):
        img = image.copy()
        img = (img+1)*127.5
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if pose is None:
            return img

        skel = pose.copy()
        skel.shape = (3, -1)
        if ratio is not None:
            skel = skel*ratio
        # skel2 = []
        # for pt in skel:
        #     # pt2 = Camera.to2D(pt+com)
        #     pt2[2] = 1.0
        #     pt2 = np.dot(trans, pt2)
        #     pt2.shape = (3, 1)
        #     pt2 = (pt2[0], pt2[1])
        #     skel2.append(pt2)
        # for idx, pt2 in enumerate(skel2):
        #     cv2.circle(img, pt2, 3,
        #                data.util.figColor[colorPlatte[idx]], -1)
        edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                 [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
        color = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0.5, 0, 0],
                 [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5],
                 [0, 0.5, 0.5], [0.5, 1, 0], [0.5, 0, 1], [0.5, 1, 1], [1, 0, 0.5]]
        for i, edge in enumerate(edges):
            pt1 = skel[0:2, edge[0]]
            pt2 = skel[0:2, edge[1]]
            cv2.line(img, (int(pt1[0]), int(pt1[1])),
                     (int(pt2[0]), int(pt2[1])), (np.asarray(color[i])*255).tolist(), 4)
        # for b in bones:
        #     pt1 = skel2[b[0]]
        #     pt2 = skel2[b[1]]
        #     color = b[2]
        #     cv2.line(img, pt1, pt2, color, 2)
        return img
