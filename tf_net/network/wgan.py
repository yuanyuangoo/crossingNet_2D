import sys
sys.path.append('./')
import globalConfig
from data.ops import *
# from data.layers import *
from data.dataset import *
from data.util import *
import tensorflow as tf
from six.moves import xrange
import shutil
from numpy.random import RandomState
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
class ImageWGAN(object):
    def __init__(self, input_height=128, input_width=128, crop=True,
                 batch_size=64, sample_num=64, output_height=128, output_width=128,
                 y_dim=15, dim_z=46, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='H36M',
                 checkpoint_dir="./checkpoint", sample_dir="samples",
                 learning_rate=0.0002, beta1=0.5, epoch=200, train_size=np.inf, reuse=False):
        self.sample_dir = os.path.join(
            globalConfig.gan_pretrain_path, sample_dir)
        self.epoch = epoch
        self.crop = crop
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.beta1 = beta1
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.train_size = train_size
        self.y_dim = y_dim
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
        # self.data_X, self.data_y = self.load_h36m()
        # self.c_dim = self.data_X[0].shape[-1]
        self.grayscale = True
        self.build_model()
        
    def build_model(self):
        #self.y is label
        self.y = tf.placeholder(
            tf.float32, [self.batch_size, self.y_dim], name='y')

        if self.crop:
            image_dims = [self.output_height, self.output_width, 1]
        else:
            image_dims = [self.input_height, self.input_width, 1]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        #real image input
        inputs = self.inputs

        #noise
        self.z = tf.placeholder(
            tf.float32, [None, self.dim_z], name='z')
        self.z_sum = histogram_summary("z", self.z)

        #Generator for fake image
        self.G = self.build_generator(self.z, self.y, reuse=False)
        #Discriminator for real image
        self.D, self.D_logits, _ = self.build_discriminator(
            inputs, self.y, reuse=False)
        #image
        self.sampler = self.build_generator(self.z, self.y, reuse=True)
        #Discriminator for fake image
        self.D_, self.D_logits_, _ = self.build_discriminator(
            self.G, self.y, reuse=True)
        self.build_metric()
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        self.d_loss_real = - tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        self.g_loss = - self.d_loss_fake
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # def sigmoid_cross_entropy_with_logits(x, y):
        #     return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        # self.smooth = 0.05
        # self.d_loss_real = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D_logits)) * (1 - self.smooth))  # for real image Discriminator
        # self.d_loss_fake = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_logits_)))  # for fake image Discriminator
        # self.g_loss = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_logits_)*(1-self.smooth)))  # for fake image Generator

        self.d_loss_real_sum = scalar_summary(
            "d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary(
            "d_loss_fake", self.d_loss_fake)



        self.lambd = 0.25
        self.disc_iters = 1
        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.inputs.get_shape(), minval=0.,maxval=1.)
        differences = self.G - self.inputs # This is different from MAGAN
        interpolates = self.inputs + (alpha * differences)
        _, D_inter, _ = self.build_discriminator(
            interpolates, self.y, reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.lambd * gradient_penalty


        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.saver = tf.train.Saver()

    def build_discriminator(self, image, y=None, reuse=False, is_training=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            self.dis_render_layer = image
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0_0 = lrelu(
                conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))

            h0 = conv_cond_concat(h0_0, yb)
            self.dis_hidden = h0_0
            self.dis_metric = h0_0
            h1_0 = lrelu(bn(
                conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'), is_training=is_training, scope="d_h1_bn"))
            self.feamat_layer = h1_0
            h1 = tf.reshape(h1_0, [self.batch_size, -1])
            h1 = concat([h1, y], 1)

            h2 = lrelu(bn(linear(h1, self.dfc_dim, 'd_h2_lin'),
                          is_training=is_training, scope="d_h2_bn"))
            h2 = concat([h2, y], 1)

            # logits
            h3 = linear(h2, 1, 'd_h3_lin')
            self.dis_px_layer = h3
            return tf.nn.tanh(h3), h3, h2

    def build_generator(self, z, y=None, reuse=False, is_training=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)

            h0 = lrelu(
                bn(linear(z, self.gfc_dim, 'g_h0_lin'), is_training=is_training, scope="g_h0_bn"))
            h0 = concat([h0, y], 1)

            h1 = lrelu(bn(
                linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), is_training=is_training, scope="g_h1_bn"))
            h1 = tf.reshape(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = lrelu(bn(deconv2d(
                h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), is_training=is_training, scope="g_h2_bn"))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.tanh(
                deconv2d(h2, [self.batch_size, s_h, s_w, 1], name='g_h3'))

    def build_recognition(self, output_dim=23, hidden_layer=None, reuse=False, keep_prob=1, is_training=True):
        if hidden_layer is None:
            hidden_layer = self.dis_hidden
        with tf.variable_scope("recognition") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = bn(
                conv2d(hidden_layer, self.df_dim, name='r_h0_conv'), is_training=is_training, scope='r_h0_bn')

            h1 = bn(
                conv2d(h0, self.df_dim, name='r_h1_conv'), is_training=is_training, scope='r_h1_bn')

            h2 = bn(
                conv2d(h1, self.df_dim, name='r_h2_conv'), is_training=is_training, scope='r_h2_bn')

            h2 = tf.reshape(h2, [self.batch_size, -1])

            h3 = linear(h2, output_dim, 'r_h3_lin')

            return lrelu(h3)

    def build_metric(self, output_dim=23, hidden_layer=None, reuse=False, keep_prob=1,is_training=True):
        if hidden_layer is None:
            hidden_layer = self.dis_metric
        with tf.variable_scope("metric") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = bn(
                conv2d(hidden_layer, self.df_dim, name='m_h0_conv'), is_training=is_training, scope='m_h0_bn')

            h1 = bn(
                conv2d(h0, self.df_dim, name='m_h1_conv'), is_training=is_training, scope='m_h1_bn')

            h2 = bn(
                conv2d(h1, self.df_dim, name='m_h2_conv'), is_training=is_training, scope='m_h2_bn')

            h2 = tf.reshape(h2, [self.batch_size, -1])

            hidden_dim = 512

            h3 = linear(h2, hidden_dim, 'm_h3_lin')
            h4 = linear(h3, hidden_dim*output_dim, 'm_h4_lin')
            h4 = tf.reshape(h4, [self.batch_size, hidden_dim, output_dim])
            return lrelu(h4)

    def build_metric_combi(self, output_dim=23, hidden_layer=None, reuse=False, keep_prob=1):
        if hidden_layer is None:
            hidden_layer = self.dis_metric
        with tf.variable_scope("metric") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = bn(
                conv2d(hidden_layer, self.df_dim, name='m_h0_conv'))

            h1 = bn(
                conv2d(h0, self.df_dim, name='m_h1_conv'))

            h2 = bn(
                conv2d(h1, self.df_dim, name='m_h2_conv'))

            h2 = tf.reshape(h2, [self.batch_size, -1])

            hidden_dim = 512

            h3 = linear(h2, hidden_dim, 'm_h3_lin')

            self.combi_input_layer = tf.placeholder(
                dtype=tf.float32, shape=(None, 2 * 1024), name="combi_input_layer")
            combi_metric = linear(self.combi_input_layer,
                                  2*hidden_dim*output_dim)
            combi_metric = tf.reshape(
                combi_metric, [self.batch_size, 2*hidden_dim, output_dim])

            return lrelu(h3), combi_metric
            
    def train(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        train_labels, _, train_img, test_labels, _, test_img, n_samples, total_batch = prep_data(
            train_dataset, valid_dataset, self.batch_size)

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

            self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                        self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
            self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
            self.writer = SummaryWriter(
                os.path.join(globalConfig.gan_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.ImageWGAN')

            sample_z = np.random.uniform(-1, 1,
                                         size=(self.sample_num, self.dim_z))

            sample_inputs = test_img
            sample_labels = test_labels

            counter = 1
            start_time = time.time()
            # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            # if could_load:
            #     counter = checkpoint_counter
            #     print(" [*] Load SUCCESS")
            # else:
            #     print(" [!] Load failed...")

            for epoch in xrange(self.epoch):
                batch_idxs = min(
                    len(train_img), self.train_size) // self.batch_size

                for idx in xrange(0, int(batch_idxs)):
                    batch_images = train_img[idx *
                                               self.batch_size:(idx+1)*self.batch_size]
                    batch_labels = train_labels[idx *
                                               self.batch_size:(idx+1)*self.batch_size]

                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.dim_z]) \
                        .astype(np.float32)
                    # Update D network
                    _, summary_str, errD = self.sess.run([d_optim, self.d_sum, self.d_loss],
                                                         feed_dict={
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.dim_z]) \
                        .astype(np.float32)
                    # Update G network
                    _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                                         feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, errD, errG))

                    if np.mod(counter, 640) == 1:
                        # show_all_variables()
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))
                    if np.mod(counter, 5000) == 2:
                        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = "ImageWGAN.model"
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
            self.saver.restore(
                self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\dbuild_generator+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        # for i in range(0, 20000, 20000):
        ds.loadH36M(40960, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        # for i in range(0, 20000, 20000):
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    gan = ImageWGAN()
    gan.train(ds, val_ds)
