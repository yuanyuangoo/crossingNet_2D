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
from data.layers import *
from data.dataset import *
from data.util import *

class ForwardRender(object):
    def __init__(self, dim_x,sample_dir="samples",checkpoint_dir="./checkpoint"):
        self.checkpoint_dir = os.path.join(
            globalConfig.Forward_Render_pretrain_path, checkpoint_dir)
        self.sample_dir = os.path.join(
            globalConfig.Forward_Render_pretrain_path, sample_dir)
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

        self.pose_input = self.pose_vae.x_hat

        #Train Ali
        self.real_image = self.image_gan.inputs
        self.pixel_loss = tf.losses.mean_squared_error(self.real_image, self.render)
        pixel_loss = (self.real_image-self.render)**2
        pixel_loss = tf.clip_by_value(pixel_loss, 0, self.golden_max)

        self.pixel_loss = tf.reduce_mean(pixel_loss)

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
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        #load train dataset
        train_skel = []
        train_labels = []
        train_img = []
        for frm in train_dataset.frmList:
            train_skel.append(frm.skel)
            train_labels.append(frm.label)
            train_img.append(frm.norm_img)
        #load valid dataset
        test_skel = []
        test_labels = []
        test_img = []
        for frm in valid_dataset.frmList:
            test_skel.append(frm.skel)
            test_labels.append(frm.label)
            test_img.append(frm.norm_img)
        train_skel = np.asarray(
            train_skel)/np.concatenate((128*np.ones(17*2), 60*np.ones(17)))
        train_labels = np.asarray(train_labels)
        train_img = np.asarray(train_img)
        test_skel = np.asarray(
            test_skel)/np.concatenate((128*np.ones(17*2), 60*np.ones(17)))

        test_labels = np.asarray(test_labels)
        test_img = np.asarray(test_img)
        #normalize skel data
        # train_skel = train_skel/max(-1*train_skel.min(), train_skel.max())
        # test_skel = test_skel/max(-1*test_skel.min(), test_skel.max())
        n_samples = train_skel.shape[0]

        seed = 42
        np_rng = RandomState(seed)
        total_batch = int(n_samples / self.batch_size)
        
        print('[ForwardRender] enter training loop with %d epoches' % nepoch)
        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()
            pose_vae_var = [val for val in tf.trainable_variables(
            ) if 'encoder' in val.name or 'decoder' in val.name]
            self.saver = tf.train.Saver(pose_vae_var)
            could_load, checkpoint_counter = self.load(
                self.pose_vae.checkpoint_dir)

            image_gan_var = [val for val in tf.trainable_variables(
            ) if 'generator' in val.name or 'discriminator' in val.name]
            self.saver = tf.train.Saver(image_gan_var)
            could_load, checkpoint_counter = self.load(
                self.image_gan.checkpoint_dir)

            self.saver = tf.train.Saver()
            if could_load:
                # counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            self.writer = SummaryWriter(
                os.path.join(globalConfig.Forward_Render_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.imageGAN')

            for epoch in tqdm(range(nepoch)):
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * self.batch_size) % (total_batch)
                    _, tot_loss = self.sess.run((self.ali_train_op, self.pixel_loss), feed_dict={
                        self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :, :, :],
                        self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                        self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],
                        self.image_gan.y: train_labels[offset:(offset + self.batch_size), :]
                    })
                    print("epoch %d: L_tot %03.2f" % (epoch, tot_loss))

                if epoch % 10 == 0:
                    samples, real = self.sess.run((self.render, self.real_image), feed_dict={
                        self.pose_input: test_skel,
                        self.pose_vae.label_hat: test_labels,
                        self.image_gan.y: test_labels,
                        self.real_image: test_img})

                    save_images(samples, image_manifold_size(samples.shape[0]),
                                '{}/train_{:02d}.png'.format(self.sample_dir, epoch), skel=test_skel)
                    save_images(real, image_manifold_size(real.shape[0]),
                                '{}/train_{:02d}_real.png'.format(self.sample_dir, epoch), skel=test_skel)

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
        skel = skel[0:2,:]
        min_s = skel.min()
        max_s = skel.max()
        mid_s = (min_s+max_s)/2
        skel = (((skel-mid_s)/(max_s-min_s))+0.52)*125
        if ratio is not None:
            skel = skel*ratio

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
        model_name = "GanRender.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)




Num_of_Joints = ref.nJoints


if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        # for i in range(0, 20000, 20000):
        ds.loadH36M(1024, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        # for i in range(0, 20000, 20000):
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    Fr = ForwardRender(dim_x=Num_of_Joints*3)
    Fr.train(100,ds, val_ds)
