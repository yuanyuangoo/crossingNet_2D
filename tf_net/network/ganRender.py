import scipy.optimize
import cv2
import time
import shutil
import os
import json
import numpy.linalg
from numpy.random import RandomState
import tensorflow as tf
from collections import namedtuple
import sys
from forwardRender import ForwardRender
import numpy as np
from tqdm import tqdm
sys.path.append('./')
sys.path.append('./data/')
from data.layers import *
from dataset import *
import globalConfig

class GanRender(ForwardRender):
    # gan=Lgan, est=Lpos, metric=Lsmo, recons=Lrecons
    DisErr = namedtuple('disErr', ['gan', 'est', 'metric'])
    GenErr = namedtuple('genErr', ['gan', 'recons', 'metric'])
    golden_max = 1.0

    def __init__(self, x_dim, rndGanInput=False, metricCombi=False, checkpoint_dir="./checkpoint", sample_dir="samples"):
        super(GanRender, self).__init__(x_dim)
        self.rndGanInput = rndGanInput
        self.metricCombi = metricCombi

        self.checkpoint_dir = os.path.join(
            globalConfig.gan_Render_pretrain_path, checkpoint_dir)
        self.sample_dir = os.path.join(
            globalConfig.gan_Render_pretrain_path, sample_dir)

        gen_vars = self.alignment_vars+self.image_gan.g_vars

        # Lrecons
        self.fake_render = self.render
        self.real_render = self.real_image
        recons_loss = (self.real_render-self.fake_render)**2
        recons_loss = tf.clip_by_value(recons_loss, 0, self.golden_max)

        self.recons_loss = tf.reduce_mean(recons_loss)

        # gan part, Lgan
        real_feamat = self.image_gan.D_logits
        real_feamat = tf.reduce_mean(real_feamat, axis=0)
        fake_feamat = self.feamat_layer
        fake_feamat = tf.reduce_mean(fake_feamat, axis=0)

        self.combi_weights_input = tf.placeholder(
            dtype=tf.float32, name='noise_combination', shape=self.latent.shape)
        # latent_noises = tf.multiply(self.combi_weights_input, self.latent)
        # aligned_gan_noise = self.alignment
        self.fake_image = self.render
        # gan_fake_image_var = tf.concat(1, [self.fake_image, self.render])

        px_fake = self.dis_px_layer
        px_real = self.image_gan.D_logits

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        loss_dis_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(px_real, tf.ones_like(px_real)) * (1 - self.image_gan.smooth))  # for real image Discriminator
        loss_dis_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(px_fake, tf.zeros_like(px_fake)))  # for fake image Discriminator
        self.loss_dis_gan = loss_dis_real+loss_dis_fake

        self.gan_loss_gen = tf.reduce_mean(abs(real_feamat-fake_feamat))

        # metric part(smooth task), (Lsmo)
        if not self.metricCombi:
            fake_metric = self.image_gan.build_metric(
                output_dim=self.z_dim, reuse=False)
            real_metric = self.image_gan.build_metric(
                output_dim=self.z_dim, reuse=True)
            self_metric = self.image_gan.build_metric(
                output_dim=self.z_dim, reuse=True)

            # latent_diff = self.latent-latent_noises
            latent_diff=self.latent
            metric_diff = real_metric+fake_metric
            self_diff = real_metric - self_metric
        else:

            fake_metric = self.image_gan.build_metric_combi(
                self.fake_image,  self.image_gan.y, output_dim=self.z_dim, reuse=False)
            real_metric = self.image_gan.build_metric_combi(
                self.image_gan.inputs,  self.image_gan.y, output_dim=self.z_dim, reuse=True)
            self_metric = self.image_gan.build_metric_combi(
                self.render, self.image_gan.y, output_dim=self.z_dim, reuse=True)

            # latent_diff = self.latent-latent_noises
            latent_diff=self.latent
            real_fake_combi = tf.concat(1, [real_metric, fake_metric])
            metric_diff = self.image_gan.build_metric_combi(
                real_fake_combi, self.image_gan.y, output_dim=self.z_dim, reuse=False)
            self_combi = tf.concat(1, [real_metric, self_metric])
            self_diff = self.image_gan.build_metric_combi(
                self_combi, self.image_gan.y, output_dim=self.z_dim, reuse=True)

        metric_loss = (latent_diff - metric_diff)**2 + self_diff**2
        self.metric_loss = tf.reduce_mean(metric_loss)

        self.gen_loss = self.gan_loss_gen + self.recons_loss + self.metric_loss
        self.gan_loss_gen_sum = scalar_summary(
            "gan_loss_gen", self.gan_loss_gen)
        self.recons_loss_sum = scalar_summary("recons_loss", self.recons_loss)
        self.metric_loss_sum = scalar_summary("metric_loss", self.metric_loss)
        self.gen_loss_sum = scalar_summary("gen_loss", self.gen_loss)

        self.gen_optim = tf.train.AdamOptimizer(
            learning_rate=self.lr, beta1=self.b1).minimize(self.gen_loss, var_list=gen_vars)
        print('gen_train_fn compiled')

        # alignment part
        self.align_optim = tf.train.AdamOptimizer(
            learning_rate=self.lr*10, beta1=self.b1).minimize(self.recons_loss, var_list=self.alignment_vars)
        print('alignment_train_fn compiled')

        # estimating the latent variable part, Lpos
        self.z_est = self.image_gan.build_recognition(
            self.image_gan.inputs, self.image_gan.y, output_dim=self.z_dim, is_training=is_training, reuse=False)
        self.z_est_t = self.image_gan.build_recognition(
            self.image_gan.inputs, self.image_gan.y, output_dim=self.z_dim, is_training=is_training, reuse=True)

        # disciminator loss on same image input consistent
        self.loss_dis_est = tf.losses.mean_squared_error(
            self.z_est_t, self.latent)

        self.dis_loss = self.loss_dis_gan + self.loss_dis_est + self.metric_loss
        self.loss_dis_gan_sum = scalar_summary(
            "loss_dis_gan", self.loss_dis_gan)
        self.loss_dis_est_sum = scalar_summary(
            "loss_dis_est", self.loss_dis_est)
        self.dis_loss_sum = scalar_summary("dis_loss", self.dis_loss)

        self.g_sum = merge_summary(
            [self.gen_loss_sum, self.gan_loss_gen_sum, self.recons_loss_sum, self.metric_loss_sum])
        self.d_sum = merge_summary(
            [self.dis_loss_sum, self.loss_dis_gan_sum, self.loss_dis_est_sum, self.metric_loss_sum])
        t_vars = tf.global_variables()
        self.image_gan.m_vars = [var for var in t_vars if 'm_' in var.name]
        self.image_gan.r_vars = [var for var in t_vars if 'r_' in var.name]
        self.dis_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.b1).minimize(
            self.dis_loss, var_list=self.image_gan.d_vars+self.image_gan.r_vars+self.image_gan.m_vars)
        print('dis_train_fn compiled')

        # initialize the training of recognition, metric part
        self.init_dis_loss = self.loss_dis_est+self.metric_loss
        self.init_dis_loss_sum = scalar_summary(
            "init_dis_loss", self.init_dis_loss)
        self.init_sum = merge_summary(
            [self.init_dis_loss_sum, self.loss_dis_est_sum, self.metric_loss_sum])
        self.init_dis_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.b1).minimize(
            self.init_dis_loss, var_list=self.image_gan.r_vars+self.image_gan.m_vars)
        print('init_dis_fn compiled')

        self.est_pose_z = tf.placeholder(dtype=tf.float32, name="est_pose_z")
        self.est_pose_t = self.pose_vae.sampler(
            self.est_pose_z, self.pose_vae.dim_x, self.pose_vae.n_hidden)
        # self.saver = tf.train.Saver()

    def train(self, nepoch=None, train_dataset=None, valid_dataset=None, desc='dummy'):
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

        print('[ganRender] begin training loop')
        seed = 42
        np_rng = RandomState(seed)
        total_batch = int(n_samples / self.batch_size)

        with tf.Session() as self.sess:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            pose_vae_var = [val for val in tf.global_variables(
            ) if 'encoder' in val.name or 'decoder' in val.name]
            self.saver = tf.train.Saver(pose_vae_var)
            could_load, checkpoint_counter = self.load(
                self.pose_vae.checkpoint_dir)

            image_gan_var = [val for val in tf.global_variables(
            ) if 'generator' in val.name or 'discriminator' in val.name]
            self.saver = tf.train.Saver(image_gan_var)
            could_load, checkpoint_counter = self.load(
                self.image_gan.checkpoint_dir)

            self.saver = tf.train.Saver()
            counter = 1
            # could_load = False
            if could_load:
                # counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            self.writer = SummaryWriter(
                os.path.join(globalConfig.gan_Render_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.imageGAN')
            for epoch in tqdm(range(nepoch)):
                gen_errs, dis_errs = np.zeros((3,)), np.zeros((3,))
                gen_err, dis_err, nupdates = 0, 0, 0
                for i in tqdm(range(total_batch)):
                    offset = (i * self.batch_size) % (n_samples)
                    combi_noises = np_rng.normal(0, 0.05,
                                                 (self.batch_size, self.pose_z_dim)
                                                 ).astype(np.float32)
                    gan_noises = GanRender.rndCvxCombination(np_rng,
                                                             src_num=self.batch_size,
                                                             tar_num=self.batch_size,
                                                             sel_num=5)
                    if epoch < 23:
                        gen_err, summary_str, _ = self.sess.run(
                            [self.recons_loss, self.recons_loss_sum, self.align_optim], feed_dict={
                                self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :, :, :],
                                self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                                self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],
                                # self.origin_input: None,
                                self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                                self.combi_weights_input: combi_noises
                                # self.pose_vae.noise_input: vae_noises,
                            })

                        self.writer.add_summary(summary_str, counter)
                        gen_errs += np.array([0., 1., 0.])*gen_err

                        est_err, metr_err, summary_str, _ = self.sess.run(
                            [self.loss_dis_est, self.metric_loss, self.init_sum, self.init_dis_optim], feed_dict={
                                self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :],
                                self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                                self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                                # self.origin_input: None,
                                self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                                self.combi_weights_input: combi_noises
                                # self.pose_vae.noise_input_var: vae_noises,
                                # self.image_gan.noise_input: gan_noises,
                            })
                        self.writer.add_summary(summary_str, counter)
                        dis_errs += np.array([0., 1., 0.])*est_err +\
                            np.array([0., 0., 1.])*metr_err
                        nupdates += 1
                        counter += 1
                        continue

                    if self.rndGanInput:
                        _, loss_dis_gan, loss_dis_est, metric_loss, summary_str, _ = self.sess.run(
                            [self.dis_loss, self.loss_dis_gan, self.loss_dis_est,
                             self.metric_loss, self.dis_loss_sum, self.dis_optim], feed_dict={
                                self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :],
                                self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                                self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                                # self.origin_input: None,
                                self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                                self.combi_weights_input: combi_noises
                                # self.pose_vae.noise_input_var: vae_noises,
                                # self.image_gan.noise_input: gan_noises,

                            })
                        self.writer.add_summary(summary_str, counter)
                        dis_errs += np.array([1., 0., 0.])*loss_dis_gan+np.array([0., 1., 0.])*loss_dis_est +\
                            np.array([0., 0., 1.])*metric_loss

                        _, gan_loss_gen, recons_loss, metric_loss, summary_str, _ = self.sess.run(
                            [self.gen_loss, self.gan_loss_gen, self.recons_loss,
                             self.metric_loss, self.gen_loss_sum, self.gen_optim], feed_dict={
                                self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :],
                                self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                                self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],
                                # self.origin_input: None,
                                self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                                self.combi_weights_input: combi_noises
                                # self.pose_vae.noise_input_var: vae_noises,
                                # self.image_gan.noise_input: gan_noises,
                            })
                        self.writer.add_summary(summary_str, counter)
                        gen_errs += np.array([1., 0., 0.])*gan_loss_gen+np.array([0., 1., 0.])*recons_loss +\
                            np.array([0., 0., 1.])*metric_loss
                    else:
                        _, loss_dis_gan, loss_dis_est, metric_loss, summary_str ,_ = self.sess.run(
                            [self.dis_loss, self.loss_dis_gan, self.loss_dis_est,
                             self.metric_loss, self.dis_loss_sum, self.dis_optim], feed_dict={
                                self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :],
                                self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                                self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                                # self.origin_input: None,
                                self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                                self.combi_weights_input: combi_noises
                                # self.pose_vae.noise_input_var: vae_noises,
                                # self.image_gan.noise_input: None,
                            })
                        self.writer.add_summary(summary_str, counter)
                        dis_errs += np.array([1., 0., 0.])*loss_dis_gan+np.array([0., 1., 0.])*loss_dis_est +\
                            np.array([0., 0., 1.])*metric_loss

                        _gan_loss_gen, recons_loss, metric_loss, summary_str, _ = self.sess.run(
                            [self.gen_loss, self.gan_loss_gen, self.recons_loss,
                             self.metric_loss, self.gen_loss_sum, self.gen_optim], feed_dict={
                                self.image_gan.inputs: train_img[offset:(offset + self.batch_size), :],
                                self.pose_vae.x_hat: train_skel[offset:(offset + self.batch_size), :],
                                self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                                # self.origin_input: None,
                                self.image_gan.y: train_labels[offset:(offset + self.batch_size), :],
                                self.combi_weights_input: combi_noises
                                # self.pose_vae.noise_input_var: vae_noises,
                                # self.image_gan.noise_input: None,
                            })
                        self.writer.add_summary(summary_str, counter)
                        gen_errs += np.array([1., 0., 0.])*gan_loss_gen+np.array([0., 1., 0.])*recons_loss +\
                            np.array([0., 0., 1.])*metric_loss
                    counter += 1
                    nupdates += 1

                dis_errs /= nupdates
                gen_errs /= nupdates
                dis_errs = self.DisErr(*tuple(dis_errs))
                gen_errs = self.GenErr(*tuple(gen_errs))
                print('epoch: {}, batch: {}'.format(epoch, i))
                print('disErr: {}'.format(dis_errs))
                print('genErr: {}'.format(gen_errs))

                n_samples_test = test_img.shape[0]
                total_batch_test = int(n_samples_test/self.batch_size)
                if epoch % 10 == 0 and valid_dataset is not None:
                    # print('epoch: {}, batch: {}'.format(epoch, n_samples))
                    # print('disErr: {}'.format(dis_errs))
                    # print('genErr: {}'.format(gen_errs))
                    for i in tqdm(range(total_batch_test)):
                        # noise = np.zeros((1, self.pose_z_dim), np.float32)
                        offset = (i * self.batch_size) % (n_samples_test)
                        reco_image = self.render.eval({
                            self.pose_input: test_skel[offset:(offset + self.batch_size), :],
                            self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                            # self.pose_vae.y: test_labels[offset+i, :],
                            # self.origin_input: None,
                            self.image_gan.y: test_labels[offset:(offset + self.batch_size), :],
                        })
                        reco_pose = self.pose_vae.y.eval({
                            self.pose_input: test_skel[offset:(offset + self.batch_size), :],
                            self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                            # self.origin_input: None,
                            # self.pose_vae.label: test_labels[offset+i, :],
                        })

                        real_pose = self.resumePose(
                            test_skel[offset], self.origin_input)

                        real_img = self.visPair(
                            test_img[offset], real_pose, self.origin_input, 50.0)

                        reco_pose = self.resumePose(
                            reco_pose[0], self.origin_input)

                        fake_img = self.visPair(
                            reco_image[0], reco_pose, self.origin_input, 50.0)

                        est_z = self.z_est_t.eval(
                            {
                                self.image_gan.inputs: test_img[offset:(offset + self.batch_size), :],
                                self.image_gan.y: test_labels[offset:(
                                    offset + self.batch_size), :]
                            })
                        est_z.shape = (self.batch_size, self.dim_z)
                        # est_z, est_orig = est_z[:20], est_z[20:]
                        # est_z.shape = (1, 20)
                        # est_orig.shape = (1, 3)
                        # est_orig=None
                        est_pose = self.est_pose_t.eval(
                            {self.est_pose_z: est_z})
                        est_image = self.render.eval({
                            self.pose_input: est_pose,
                            self.pose_vae.label_hat: train_labels[offset:(offset + self.batch_size), :],

                            # self.pose_vae.y: test_labels[offset:(offset + self.batch_size), :],
                            # self.origin_input: None,
                            self.image_gan.y: test_labels[offset:(offset + self.batch_size), :],
                        })
                        est_pose = self.resumePose(
                            est_pose[0], self.origin_input)
                        est_img = self.visPair(est_image[0],
                                                est_pose,
                                                self.origin_input,
                                                50.0)
                        recons_image = np.hstack(
                            (real_img, fake_img, est_img))
                        cv2.imwrite(os.path.join(self.sample_dir, '%d_%d.jpg' % (epoch, i)),
                                    recons_image.astype('uint8'))

                if epoch % 100 == 0:
                    self.save(self.checkpoint_dir, epoch)

    @classmethod
    def rndCvxCombination(cls, rng, src_num, tar_num, sel_num):
        # generate tar_num random convex combinations from src_num point, every
        # time only sel_num from the src_num are used
        if sel_num > src_num:
            raise ValueError('sel_num %d should less then src_num %d' % (sel_num,
                                                                         src_num))
        m = np.zeros((tar_num, src_num))
        for s in m:
            sel = rng.choice(src_num, sel_num)
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
    @property
    def model_dir(self):
        return "{}_{}".format(
            globalConfig.dataset, self.batch_size,
        )

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


if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        for i in range(0, 20000, 20000):
            ds.loadH36M(i, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadH36M(i, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    print('validation length = %d' % len(val_ds.frmList))
    skel_num = len(val_ds.frmList[0].skel)
    print('skel_num=%d' % skel_num)
    render = GanRender(x_dim=skel_num, rndGanInput=True, metricCombi=False)

    desc = 'pretrained'
    render.train(1001, ds, val_ds,
                 desc=desc)
    # render.test(val_ds, desc=desc, modelIdx='-1')
