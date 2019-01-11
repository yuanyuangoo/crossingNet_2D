import os
import time
from six.moves import xrange
import tensorflow as tf

import sys
sys.path.append('./')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data.util import *
from data.dataset import *
from data.layers import *
import globalConfig

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class imagegan(object):
    def __init__(self, batch_size=64, output_height=128, output_width=128, label_dim=15, dim_z=17*3, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='H36M', checkpoint_dir="./checkpoint", sample_dir="samples",
                 learning_rate=0.0002, beta1=0.5, epoch=300, reuse=False):
        self.sample_dir = os.path.join(
            globalConfig.gan_pretrain_path, sample_dir)
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta1 = beta1
        self.output_height = output_height
        self.output_width = output_width
        self.label_dim = label_dim
        self.dim_z = dim_z
        self.c_dim = 1
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.dataset_name = dataset_name
        self.checkpoint_dir = os.path.join(
            globalConfig.gan_pretrain_path, checkpoint_dir)
        self.build_model()

    def build_model(self):
        #self.y is label
        self.y = tf.placeholder(
            tf.float32, [self.batch_size, self.label_dim], name='y')

        self.pose_input = tf.placeholder(
            tf.float32, [self.batch_size, self.dim_z], name='pose_input')

        #real image input
        self.image_target = tf.placeholder(
            tf.float32, [self.batch_size, self.output_height, self.output_width, 1], name='background_image')
        self.image_target_sum = image_summary(
            "image_target", self.image_target)

        #Generator for fake image
        self.G = self.build_generator(
            self.pose_input, self.y, is_training=True, reuse=False)

        #image
        self.sample = self.build_generator(
            self.pose_input, self.y,  reuse=True, is_training=False)

        # self.build_metric()
        self.G_sum = image_summary("G", self.G)

        self.g_loss_l1 = tf.reduce_mean(tf.abs(self.image_target - self.G))*10
        self.g_loss_l2 = tf.nn.l2_loss(self.image_target - self.G)

        self.D_real = self.build_discriminator(
            self.image_target, reuse=False, is_training=True)
        self.D_fake = self.build_discriminator(
            self.G, reuse=True, is_training=True)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        self.smooth = 0.05
        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)) * (1 - self.smooth))  # for real image Discriminator
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(self.D_fake)))  # for fake image Discriminator
        self.g_loss_cross = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_fake, tf.ones_like(self.D_fake)*(1-self.smooth)))  # for fake image Generator

        self.d_loss_real_sum = scalar_summary(
            "d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary(
            "d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = self.g_loss_cross+self.g_loss_l1
        self.g_loss_cross_sum = scalar_summary(
            "g_loss_cross", self.g_loss_cross)
        self.g_loss_l1_sum = scalar_summary("g_loss_l1", self.g_loss_l1)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.global_variables()

        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.saver = tf.train.Saver()

        self.g_sum = merge_summary(
            [self.G_sum, self.g_loss_sum, self.g_loss_l1_sum, self.g_loss_cross_sum])
        self.d_sum = merge_summary(
            [self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])

    def build_discriminator(self, image, y=None, reuse=False, is_training=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # self.dis_render_layer = image
            # yb = tf.reshape(y, [self.batch_size, 1, 1, self.label_dim])
            # x = conv_cond_concat(image, yb)

            h0 = lrelu(
                conv2d(image, self.df_dim, name='d_h0_conv'))

            h1 = lrelu(bn(
                conv2d(h0, self.df_dim, name='d_h1_conv'), is_training=is_training, scope="d_h1_bn"))

            # self.feamat_layer = h1

            h1 = tf.reshape(h1, [self.batch_size, -1])

            h2 = lrelu(bn(linear(h1, self.dfc_dim, 'd_h2_lin'),
                          is_training=is_training, scope="d_h2_bn"))

            # logits
            h3 = linear(h2, 1, 'd_h3_lin')

            return h3

    def build_generator(self, z, y=None, reuse=False, is_training=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.label_dim])
            z = concat([z, y], 1)

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)

            h0 = lrelu(
                bn(dropout(linear(z, self.gfc_dim, 'g_h0_lin'), is_training=is_training), is_training=is_training, scope="g_h0_bn"))

            # h0 = concat([h0, y], 1)

            h1 = lrelu(bn(
                dropout(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), is_training=is_training), is_training=is_training, scope="g_h1_bn"))
            h1 = tf.reshape(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            # h1 = conv_cond_concat(h1, yb)

            h2 = lrelu(bn(deconv2d(
                h1, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h2'), is_training=is_training, scope="g_h2_bn"))

            h3 = tf.nn.tanh(
                deconv2d(h2, [self.batch_size, s_h, s_w, 1], name='g_h3'))

            # h4 = tf.nn.tanh(bn(
            #     linear(tf.layers.flatten(h3), h3.shape[1]*h3.shape[2]*h3.shape[3], "g_h4_lin"), is_training=is_training, scope='g_h4_bn'))

            render = lrelu(
                conv2d(h3, h3.shape[3], name='g_conv'))

            h4 = tf.nn.tanh(bn(
                linear(tf.layers.flatten(render), render.shape[1]*render.shape[2]*render.shape[3], "g_h4_lin"), is_training=is_training, scope='g_h4_bn'))

            h4 = tf.reshape(
                h4, [self.batch_size, int(self.output_height/2), int(self.output_width/2), 1])

            h4 = tf.nn.tanh(
                deconv2d(h4, [self.batch_size, s_h, s_w, 1], name='g_h4'))
            return h4

    def test(self, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        test_labels, test_skel, test_img, _, _, _ = prep_data(
            valid_dataset, self.batch_size)
        with tf.Session() as self.sess:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(self.g_loss, var_list=self.g_vars)

            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            sample_inputs = test_img
            sample_labels = test_labels
            sample_pose = test_skel

            counter = 1
            start_time = time.time()
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            samples = self.sess.run(
                self.sample,
                feed_dict={
                    self.pose_input: sample_pose,
                    self.image_target: sample_inputs,
                    self.y: sample_labels,
                }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                        '{}/test.png'.format(self.sample_dir), skel=test_skel)

            save_images(test_img, image_manifold_size(test_img.shape[0]),
                        '{}/test_real.png'.format(self.sample_dir), skel=test_skel)

    def train(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        train_labels, train_skel, train_img,  _, n_samples, total_batch = prep_data(
            train_dataset, self.batch_size, with_background=True)
        test_labels, test_skel, test_img, _, _, _ = prep_data(
            valid_dataset, self.batch_size, with_background=True)

        train_skel = np.random.rand(train_skel.shape[0], train_skel.shape[1])
        test_skel = np.random.rand(test_skel.shape[0], train_skel.shape[1])

        with tf.Session() as self.sess:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(self.d_loss, var_list=self.d_vars)

                g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(self.g_loss, var_list=self.g_vars)

            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            self.writer = SummaryWriter(
                os.path.join(globalConfig.gan_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.imagegan')

            sample_inputs = test_img[0:64]
            sample_labels = test_labels[0:64]
            sample_pose = test_skel[0:64]

            counter = 1
            start_time = time.time()

            could_load, checkpoint_counter = self.load(self.checkpoint_dir)

            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            errD = 1.1
            errG = 0.8
            

            epoch_last=counter//total_batch
            for epoch in range(epoch_last,self.epoch):
                batch_idxs = len(train_img) // self.batch_size

                for idx in range(int(batch_idxs)):
                    batch_images = train_img[idx *
                                             self.batch_size:(idx+1)*self.batch_size]
                    batch_labels = train_labels[idx *
                                                self.batch_size:(idx+1)*self.batch_size]

                    batch_pose = train_skel[idx *
                                            self.batch_size:(idx+1)*self.batch_size]
                    if errD > 1:
                        # Update D network
                        _, summary_str, errD, errG = self.sess.run([d_optim,  self.d_sum,  self.d_loss, self.g_loss],
                                                                feed_dict={
                            self.pose_input: batch_pose,
                            self.y: batch_labels,
                            self.image_target: batch_images
                        })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str, errD, errG = self.sess.run([g_optim, self.g_sum, self.d_loss, self.g_loss],
                                                               feed_dict={
                        self.pose_input: batch_pose,
                        self.y: batch_labels,
                        self.image_target: batch_images
                    })
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f,err_D g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, errD, errG))

                    if np.mod(counter, total_batch) == 1:
                        samples = self.sess.run(
                            self.sample,
                            feed_dict={
                                self.pose_input: sample_pose,
                                # self.image_target: sample_inputs,
                                self.y: sample_labels
                            }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx), skel=None)

                    if np.mod(counter, total_batch*10) == 2:
                        self.save(self.checkpoint_dir, counter)

            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = "imagegan.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        ds.loadH36M(64*200, mode='train', tApp=True,
                    replace=False, with_background=True)

        val_ds = Dataset()
        val_ds.loadH36M(64, mode='valid', tApp=True,
                        replace=False, with_background=True)
        gan = imagegan(dim_z=17*3)

    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(64*100, mode='train', tApp=True, replace=True)

        val_ds = Dataset()
        val_ds.loadApe(64, mode='valid', tApp=True, replace=True)

        gan = imagegan(dim_z=15*3, label_dim=7)

    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    gan.train(ds, val_ds)
    # gan.test(val_ds)
