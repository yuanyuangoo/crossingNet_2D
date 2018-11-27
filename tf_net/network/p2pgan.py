import sys
sys.path.append('./')
sys.path.append('./data/')
import globalConfig
from data.dataset import *
from data.util import *
import tensorflow as tf
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from data.layers import *
EPS = 1e-12
CROP_SIZE = 128
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class P2PGAN(object):
    def __init__(self, mode='train', sample_dir='samples', checkpoint_dir="./checkpoint",
                 epoch=200, aspect_ratio=1, batch_size=64, ngf=64, ndf=64, scale_size=286, train_size=4000, test_size=1000, flip=True, label_dim=15, learning_rate=0.0002,
                 beta1=0.5, l1_weight=100, gan_weight=1, dataset_name='H36M', input_width=128, input_height=128):
        self.sample_dir = os.path.join(
            globalConfig.p2p_pretrain_path, sample_dir)
        self.epoch = epoch
        self.train_size = train_size
        self.test_size = test_size

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta1 = beta1
        self.checkpoint_dir = os.path.join(
            globalConfig.p2p_pretrain_path, checkpoint_dir)
        self.aspect_ratio = aspect_ratio
        self.ngf = ngf
        self.ndf = ndf
        self.scale_size = scale_size
        self.flip = flip
        self.l1_weight = l1_weight
        self.gan_weight = gan_weight
        self.dataset_name = dataset_name
        self.input_width = input_width
        self.input_height = input_height
        self.label_dim = label_dim
        self.build_model()

    def build_model(self):
        with tf.variable_scope("p2p") as scope:
            #real image
            self.image_target = tf.placeholder(
                tf.float32, [self.batch_size, self.input_width, self.input_height, None], name='real_image')
            #edge image
            self.image_input = tf.placeholder(
                tf.float32, [self.batch_size, self.input_width, self.input_height, None], name='background_image')
            #label
            self.label = tf.placeholder(
                tf.float32, [self.batch_size.self.label_dim], name='label')
            self.image_target_sum = histogram_summary(
                "image_target", self.image_target)
            self.image_input_sum = histogram_summary(
                "image_input", self.image_input)
            self.label_sum = histogram_summary(
                "label", self.label)
            out_channels = self.image_target.shape[-1]
            self.G = self.build_generator(self.image_input, out_channels)
            #real D
            self.D = self.build_discriminator(
                self.image_input, self.image_target, reuse=False)
            #fake D
            self.D_ = self.build_discriminator(
                self.image_input, self.G, reuse=True)
            self.sampler = self.build_generator(self.image_input, out_channels)

            self.d_sum = histogram_summary("d", self.D)
            self.d__sum = histogram_summary("d_", self.D_)
            self.G_sum = image_summary("G", self.G)

            self.g_loss_GAN = tf.reduce_mean(-tf.log(self.D_ + EPS))
            self.g_loss_L1 = tf.reduce_mean(tf.abs(self.image_target - self.G))
            self.g_loss = self.g_loss_GAN * self.gan_weight + self.g_loss_L1 * self.l1_weight
            self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

            self.d_loss = tf.reduce_mean(-(tf.log(self.D + EPS) +
                                           tf.log(1 - self.D_ + EPS)))
            self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

            t_vars = tf.trainable_variables()
            self.d_vars = [
                var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            self.saver = tf.train.Saver()

    def train(self, train_dataset, valid_dataset):
        with tf.Session() as self.sess:
            d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            self.g_sum = merge_summary([self.image_input_sum, self.d__sum,
                                        self.G_sum, self.g_loss_sum])
            self.d_sum = merge_summary(
                [self.image_input_sum, self.image_target_sum, self.d_sum,  self.d_loss_sum])
            self.writer = SummaryWriter(
                os.path.join(globalConfig.p2p_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.p2pGAN')

            counter = 1
            start_time = time.time()
            batch_idxs = int(nsamples/self.batch_size)
            for epoch in xrange(self.epoch):
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

                    errD = self.d_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, errD, errG))

                    if np.mod(counter, 100) == 1:
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
                    if np.mod(counter, 500) == 2:
                        self.save(self.checkpoint_dir, counter)

    def build_discriminator(self, input, target, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            n_layers = 3
            layers = []
            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([input, target], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = discrim_conv(input, self.ndf, stride=2)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = discrim_conv(
                        layers[-1], out_channels, stride=stride)
                    normalized = batchnorm(convolved)
                    rectified = lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

    def build_generator(self, input, out_channels):
        with tf.variable_scope("discriminator") as scope:
            layers = []
            with tf.variable_scope("encoder_1"):
                output = gen_conv(input, self.ngf)
                layers.append(output)

            layer_specs = [
                # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                self.ngf * 2,
                # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                self.ngf * 4,
                # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                self.ngf * 8,
                # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                self.ngf * 8,
                # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                self.ngf * 8,
                # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                self.ngf * 8,
                # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
                self.ngf * 8,
            ]

            for out_channels in layer_specs:
                with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                    rectified = lrelu(layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = gen_conv(rectified, out_channels)
                    output = batchnorm(convolved)
                    layers.append(output)

            layer_specs = [
                # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (self.ngf * 8, 0.5),
                # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (self.ngf * 8, 0.5),
                # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (self.ngf * 8, 0.5),
                # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (self.ngf * 8, 0.0),
                # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (self.ngf * 4, 0.0),
                # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (self.ngf * 2, 0.0),
                # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
                (self.ngf, 0.0),
            ]
            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = layers[-1]
                    else:
                        input = tf.concat(
                            [layers[-1], layers[skip_layer]], axis=3)

                    rectified = tf.nn.relu(input)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = gen_deconv(rectified, out_channels)
                    output = batchnorm(output)

                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    layers.append(output)

            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("decoder_1"):
                input = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = tf.nn.relu(input)
                output = gen_deconv(rectified, out_channels)
                output = tf.tanh(output)
                layers.append(output)

            return layers[-1]

        @property
        def model_dir(self):
            return "{}_{}".format(
                self.dataset_name, self.batch_size)

        def save(self, checkpoint_dir, step):
            model_name = "p2pGAN.model"
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


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                          1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


if __name__ == '__main__':
    # if globalConfig.dataset == 'H36M':
    #     import data.h36m as h36m
    #     ds = Dataset()
    #     for i in range(0, 20000, 20000):
    #         ds.loadH36M(i, mode='train', tApp=True, replace=False)

    #     val_ds = Dataset()
    #     for i in range(0, 20000, 20000):
    #         val_ds.loadH36M(i, mode='valid', tApp=True, replace=False)
    # else:
    #     raise ValueError('unknown dataset %s' % globalConfig.dataset)
    background_dir = os.path.join(globalConfig.h36m_base_path, "Background")
    real_dir = os.path.join(globalConfig.h36m_base_path, "images")
    train_input=[]
    train_target=[]
    with open(os.path.join(globalConfig.h36m_base_path, "annot", "train_images.txt"), 'r') as f:
        for line in f:
            background = cv2.resize(cv2.imread(
                os.path.join(background_dir, line[:-1]), 1), (128, 128))
            real = cv2.resize(cv2.imread(
                os.path.join(real_dir, line[:-1]), 1), (128, 128))
            background = np.asarray(background-127.5, np.float32)/127.5
            real = np.asarray(real-127.5, np.float32)/127.5
            train_input.append(background)
            train_target.append(real)


    p2p = P2PGAN()
    # p2p.train(ds, val_ds)
