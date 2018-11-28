import sys
sys.path.append('./')
import globalConfig
from data.layers import *
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

class ImageGAN(object):
    def __init__(self, input_height=128, input_width=128, crop=True,
                 batch_size=64, sample_num=64, output_height=128, output_width=128,
                 y_dim=15, dim_z=46, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='H36M',
                 checkpoint_dir="./checkpoint", sample_dir="samples",
                 learning_rate=0.0002, beta1=0.5, epoch=200, train_size=np.inf, reuse=False):
        self.sample_dir = os.path.join(
            globalConfig.gan_pretrain_path, sample_dir, str(dim_z)+"")
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
        #image
        self.sampler = self.build_generator(self.z, self.y, reuse=True)

        #Discriminator for real image
        self.D_logits, y_real = self.build_discriminator(
            inputs, self.y,  reuse=False)
        #Discriminator for fake image
        self.D_logits_, y_fake = self.build_discriminator(
            self.G, self.y, reuse=True)
        
        self.d_sum = histogram_summary("d", self.D_logits)
        self.d__sum = histogram_summary("d_", self.D_logits_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        self.smooth = 0.05
        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D_logits)) * (1 - self.smooth))  # for real image Discriminator
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_logits_)))  # for fake image Discriminator
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_logits_)*(1-self.smooth)))  # for fake image Generator

        self.loss_cls_real = tf.losses.mean_squared_error(self.y, y_real)
        self.loss_cls_fake = tf.losses.mean_squared_error(self.y, y_fake)

        self.loss_cls_fake_sum = scalar_summary(
            "loss_cls_fake", self.loss_cls_fake)
        self.loss_cls_real_sum = scalar_summary(
            "loss_cls_real", self.loss_cls_real)
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake + \
            0*(self.loss_cls_real+self.loss_cls_fake)
        self.g_loss = self.g_loss+0*self.loss_cls_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.saver = tf.train.Saver()

    def build_discriminator(self, image,  y=None, keep_prob=0.9, reuse=False):
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            input = image

            self.dis_render = image

            dis_conv1 = batchnorm(lrelu(conv2d(input)))
            dis_dropout1 = conv_cond_concat(
                tf.nn.dropout(dis_conv1, keep_prob), yb)

            dis_conv2 = batchnorm(lrelu(conv2d(dis_dropout1)))
            dis_dropout2 = conv_cond_concat(
                tf.nn.dropout(dis_conv2, keep_prob), yb)

            self.dis_hidden = dis_conv2
            self.dis_metric = dis_conv2

            dis_conv3 = batchnorm(lrelu(conv2d(dis_dropout2)))
            dis_dropout3 = conv_cond_concat(
                tf.nn.dropout(dis_conv3, keep_prob), yb)

            dis_conv4 = batchnorm(lrelu(conv2d(dis_dropout3)))
            dis_dropout4 = conv_cond_concat(
                tf.nn.dropout(dis_conv4, keep_prob), yb)

            dis_conv5 = batchnorm(lrelu(conv2d(dis_dropout4)))
            dis_dropout5 = conv_cond_concat(
                tf.nn.dropout(dis_conv5, keep_prob), yb)
            self.feamat = dis_conv5

            Y_ = tf.layers.dense(tf.reshape(
                dis_dropout5, (self.batch_size, -1)), units=self.y_dim)
            dis_full = tf.layers.dense(
                tf.reshape(dis_dropout5, (self.batch_size, -1)))
            self.dis_px = dis_full

            return dis_full,Y_

    def build_generator(self, z, y=None, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            input = concat([z, y], 1)

            noise_input = tf.nn.tanh(input)
            dense_1 = batchnorm(
                lrelu(tf.layers.dense(noise_input, 32*4*4)), axis=1)
            deconv_0 = tf.reshape(
                dense_1, [self.batch_size, 4, 4, 32])
            deconv_1 = conv_cond_concat(batchnorm(lrelu(deconv(deconv_0))), yb)
            deconv_2 = conv_cond_concat(batchnorm(lrelu(deconv(deconv_1))), yb)
            deconv_3 = conv_cond_concat(batchnorm(lrelu(deconv(deconv_2))), yb)
            deconv_4 = conv_cond_concat(batchnorm(lrelu(deconv(deconv_3))), yb)
            deconv_5 = tf.nn.tanh(deconv(deconv_4, out_channels=1))

            return deconv_5

    def build_recognition(self, y, output_dim, reuse=False, hidden_layer=None):
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

        if hidden_layer is None:
            hidden_layer = self.dis_hidden
        input = conv_cond_concat(hidden_layer, yb)

        with tf.variable_scope("recognition") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = conv_cond_concat(batchnorm(lrelu(conv2d(input))), yb)
            conv2 = conv_cond_concat(batchnorm(lrelu(conv2d(conv1))), yb)
            conv3 = conv_cond_concat(batchnorm(lrelu(conv2d(conv2))), yb)
            conv3=tf.reshape(conv3,(self.batch_size,-1))
            reco_layer = batchnorm(
                lrelu(tf.layers.dense(conv3, output_dim)), axis=1)
            reco_layer = concat([reco_layer, y], 1)
            reco_layer = tf.layers.dense(reco_layer, output_dim)

            return reco_layer

    def build_metric(self, y, output_dim=512, reuse=False, hidden_layer=None):
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

        if hidden_layer is None:
            hidden_layer = self.dis_metric
        input = conv_cond_concat(hidden_layer, yb)

        with tf.variable_scope("metric") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = conv_cond_concat(batchnorm(lrelu(conv2d(input))), yb)
            conv2 = conv_cond_concat(batchnorm(lrelu(conv2d(conv1))), yb)
            conv3 = conv_cond_concat(batchnorm(lrelu(conv2d(conv2))), yb)
            conv3=tf.reshape(conv3,(self.batch_size,-1))
            metric_layer = batchnorm(
                lrelu(tf.layers.dense(conv3, output_dim)), axis=1)
            metric_layer = concat([metric_layer, y], 1)
            metric_layer = tf.layers.dense(metric_layer, self.dim_z)

            return metric_layer

    def build_metric_combi(self, y, output_dim=512, reuse=False, hidden_layer=None):
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

        if hidden_layer is None:
            hidden_layer = self.dis_metric
        input = conv_cond_concat(hidden_layer, yb)

        with tf.variable_scope("metric") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = conv_cond_concat(batchnorm(lrelu(conv2d(input))), yb)
            conv2 = conv_cond_concat(batchnorm(lrelu(conv2d(conv1))), yb)
            conv3 = conv_cond_concat(batchnorm(lrelu(conv2d(conv2))), yb)
            conv3=tf.reshape(conv3,(self.batch_size,-1))
            metric_layer = batchnorm(
                lrelu(tf.layers.dense(conv3, output_dim)), axis=1)
            input_layer = tf.placeholder(dtype=tf.float32, shape=(
                None, 2*output_dim), name="combi_input")

            combi_metric_layer = tf.layers.dense(
                input_layer, (2*output_dim, self.dim_z))

            return metric_layer

    def train(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        # show_all_variables()
        train_data = []
        train_labels = []

        for frm in train_dataset.frmList:
            train_data.append(frm.norm_img)
            train_labels.append(frm.label)

        test_data = []
        test_labels = []
        for frm in valid_dataset.frmList:
            test_data.append(frm.norm_img)
            test_labels.append(frm.label)
        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        # X = np.concatenate(
        #     (train_data, test_data), axis=0)
        # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

        # np.random.shuffle(X)
        # np.random.seed(seed)
        # np.random.shuffle(y)
        idx=list(range(train_data.shape[0]))
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(idx)

        self.data_X = train_data[idx]
        self.data_y = train_labels[idx]

        # idx=list(range(test_labels.shape[0]))
        # seed = 547
        # np.random.seed(seed)
        # np.random.shuffle(idx)

        self.sample_inputs=test_data
        self.sample_labels=test_labels

        with tf.Session() as self.sess:
            d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            self.g_sum = merge_summary(
                [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.loss_cls_fake_sum])
            self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.loss_cls_real_sum])

            self.writer = SummaryWriter(
                os.path.join(globalConfig.gan_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.imageGAN')

            sample_z = np.random.uniform(-1, 1,
                                         size=(self.sample_num, self.dim_z))

            # sample_inputs = self.data_X[0:self.sample_num]
            # sample_labels = self.data_y[0:self.sample_num]

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
                    len(self.data_X), self.train_size) // self.batch_size

                for idx in xrange(0, int(batch_idxs)):
                    batch_images = self.data_X[idx *
                                               self.batch_size:(idx+1)*self.batch_size]
                    batch_labels = self.data_y[idx *
                                               self.batch_size:(idx+1)*self.batch_size]

                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.dim_z]) \
                        .astype(np.float32)

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.y: batch_labels})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errD_fake_cls = self.loss_cls_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                    errD_real_cls = self.loss_cls_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })

                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake+errD_real, errG))
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, errD_fake_cls: %.8f, errD_real_cls: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake_cls, errD_real_cls))
                    if np.mod(counter, 100) == 1:
                        # show_all_variables()
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: self.sample_inputs,
                                self.y: self.sample_labels,
                            }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))
                    if np.mod(counter, 500) == 2:
                        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = "IMAGEGAN.model_"+str(self.dim_z)
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
        train_size = 4096
        test_size = 64
        # for i in range(0, tol_train_size):
        ds.loadH36M(train_size, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        # for i in range(0, 20000, 20000):
        val_ds.loadH36M(test_size, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    gan = ImageGAN()
    gan.train(ds, val_ds)
