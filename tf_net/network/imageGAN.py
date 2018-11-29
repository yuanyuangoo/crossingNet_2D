import sys
sys.path.append('./')
import globalConfig
from data.ops import *
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

class ImageGAN(object):
    def __init__(self, input_height=128, input_width=128, crop=True,
                 batch_size=64, sample_num=64, output_height=128, output_width=128,
                 y_dim=15, dim_z=46, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='H36M',
                 checkpoint_dir="./checkpoint", sample_dir="samples",
                 learning_rate=0.0002, beta1=0.5, epoch=200, train_size=np.inf, reuse=False):
        self.sample_dir = os.path.join(
            globalConfig.gan_pretrain_path, sample_dir, str(dim_z)+"_AC")
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
        _, self.D_logits, input_for_classifier = self.build_discriminator(
            inputs, reuse=False)
        #Discriminator for fake image
        _, self.D_logits_, input_for_classifier_ = self.build_discriminator(
            self.G, reuse=True)

        _, self.cls_real = self.classifier(input_for_classifier, reuse=False)
        _, self.cls_fake = self.classifier(input_for_classifier_, reuse=True)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
            self.D_logits, tf.ones_like(self.D_logits)))  # for real image Discriminator
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.zeros_like(self.D_logits_)))  # for fake image Discriminator
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.ones_like(self.D_logits_)))  # for fake image Generator

        self.cls_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.cls_real, self.y))
        self.cls_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.cls_fake, self.y))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = self.g_loss
        self.cls_loss = self.cls_loss_real + self.cls_loss_fake

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.cls_loss_real_sum = scalar_summary(
            "cls_loss_real", self.cls_loss_real)
        self.cls_loss_fake_sum = scalar_summary(
            "cls_loss_fake", self.cls_loss_fake)
        self.cls_loss_sum = scalar_summary("cls_loss", self.cls_loss)

        self.g_sum = merge_summary(
            [self.z_sum, self.d_loss_fake_sum,  self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.cls_sum = merge_summary(
            [self.cls_loss_real_sum, self.cls_loss_fake_sum, self.cls_loss_sum])

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.cls_vars = [
            var for var in t_vars if 'generator' in var.name]+self.g_vars+self.d_vars

        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        self.cls_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.cls_loss, var_list=self.cls_vars)
        self.saver = tf.train.Saver()

    def build_discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            self.dis_render = x

            net1 = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net2 = lrelu(bn(conv2d(net1, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))

            self.dis_hidden = net2
            self.dis_metric = net2

            net3 = tf.reshape(net2, [self.batch_size, -1])
            net4 = lrelu(bn(linear(net3, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net4, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)

            self.feamat = net4
            self.dis_px = out_logit
            return out, out_logit, net4

    def classifier(self, x, is_training=True, reuse=False):
        with tf.variable_scope("classifier",reuse=reuse) as scope:
            net = lrelu(bn(linear(x, 128, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            out_logit = linear(net, self.y_dim, scope='c_fc2')
            out = tf.nn.softmax(out_logit)

            return out, out_logit

    def build_generator(self, z, y=None, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse) as scope:

            # merge noise and code
            z = concat([z, y], 1)

            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'),
                                is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(
                net, 128 * 8 * 8, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.nn.relu(bn(linear(
                net, 128 * 32 * 32, scope='g_fc3'), is_training=is_training, scope='g_bn3'))
            # net = tf.nn.relu(bn(linear(
            #     net, 128 * 32 * 32, scope='g_fc4'), is_training=is_training, scope='g_bn4'))

            net = tf.reshape(net, [self.batch_size, 32, 32, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 64, 64, 64], 4, 4, 2, 2, name='g_dc5'), is_training=is_training,
                   scope='g_bn5'))

            out = tf.nn.sigmoid(
                deconv2d(net, [self.batch_size, 128, 128, 1], 4, 4, 2, 2, name='g_dc6'))

            return out

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
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            self.writer = SummaryWriter(
                os.path.join(globalConfig.gan_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.imageGAN')

            sample_z = np.random.uniform(-1, 1,
                                         size=(self.sample_num, self.dim_z))

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

                    # update D network
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                           feed_dict={self.inputs: batch_images, self.y: batch_labels,
                                                                      self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # update G & Q network
                    _, summary_str_g, g_loss, _, summary_str_cls, cls_loss = self.sess.run(
                        [self.g_optim, self.g_sum, self.g_loss,
                            self.cls_optim, self.cls_sum, self.cls_loss],
                        feed_dict={self.z: batch_z, self.y: batch_labels, self.inputs: batch_images})
                    self.writer.add_summary(summary_str_g, counter)
                    self.writer.add_summary(summary_str_cls, counter)

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f,cls: %.8f"
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss, cls_loss))

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
        model_name = "IMAGEGAN.model_AC"+str(self.dim_z)
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
        train_size = 9192
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
