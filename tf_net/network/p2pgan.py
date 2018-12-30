import sys
sys.path.append('./')
sys.path.append('./data/')
import globalConfig
from data.dataset import *
from data.util import *
from data.layers import *
import os
import cv2
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
EPS = 1e-12
CROP_SIZE = 128
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
gray2rgb = tf.image.grayscale_to_rgb
class P2PGAN(object):
    def __init__(self, mode='train', sample_dir='samples', checkpoint_dir="./checkpoint",
                 epoch=20000, aspect_ratio=1, batch_size=64, ngf=64, ndf=64, scale_size=286, train_size=4000, test_size=1000, flip=True, label_dim=15, learning_rate=0.0002,
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
                tf.float32, [self.batch_size, self.input_width, self.input_height, 3], name='real_image')

            self.image_input = tf.placeholder(
                tf.float32, [self.batch_size, self.input_width, self.input_height, 1], name='background_image')

            #label
            self.label = tf.placeholder(
                tf.float32, [self.batch_size, self.label_dim], name='label')
            self.image_target_sum = histogram_summary(
                "image_target", self.image_target)
            self.image_input_sum = histogram_summary(
                "image_input", self.image_input)
            self.label_sum = histogram_summary(
                "label", self.label)
            out_channels = self.image_target.shape[-1]
            self.G = self.build_generator(gray2rgb(self.image_input), out_channels)
            #real D
            self.D = self.build_discriminator(
                gray2rgb(self.image_input), self.image_target, reuse=False)
            #fake D
            self.D_ = self.build_discriminator(
                gray2rgb(self.image_input), self.G, reuse=True)
            self.sampler = self.build_generator(
                gray2rgb(self.image_input), out_channels,  reuse=True, is_training=False)

            self.d_sum = histogram_summary("d", self.D)
            self.d__sum = histogram_summary("d_", self.D_)
            self.G_sum = image_summary("G", self.G)

            mask = ((-1*self.image_input)+1)/2


            self.g_loss_GAN = tf.reduce_mean(-tf.log(self.D_ + EPS))

            self.g_loss_L1 = tf.reduce_mean(
                tf.abs(tf.multiply(self.image_target - self.G, mask)))

            self.g_loss = self.g_loss_GAN * self.gan_weight + self.g_loss_L1 * self.l1_weight
            self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

            self.d_loss = tf.reduce_mean(-(tf.log(self.D + EPS) +
                                           tf.log(1 - self.D_ + EPS)))
            self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

            t_vars = tf.global_variables()
            self.d_vars = [
                var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            self.saver = tf.train.Saver()

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
                rectified = lrelu(convolved)
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
                    rectified = lrelu(normalized)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

    def build_generator(self, input, generator_outputs_channels, reuse=False, is_training=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
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
                # # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
                # self.ngf * 8,
            ]

            for out_channels in layer_specs:
                with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                    rectified = lrelu(layers[-1])
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = gen_conv(rectified, out_channels)
                    output = batchnorm(convolved)
                    layers.append(output)

            layer_specs = [
                # # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                # (self.ngf * 8, 0.5),
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
                output = gen_deconv(rectified, generator_outputs_channels)
                output = tf.tanh(output)
                layers.append(output)

            return layers[-1]

    def test(self, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        test_labels, test_skel, test_img, test_img_rgb, _, _ = prep_data(
            valid_dataset, self.batch_size)
        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()

            # self.saver = tf.train.Saver()
            # could_load, checkpoint_counter = self.load(
            #     self.checkpoint_dir)

            samples, real_image = self.sess.run(
                [self.sampler, self.image_target],
                feed_dict={
                    self.image_target: test_img_rgb,
                    self.image_input: test_img,
                    self.label: test_labels
                }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                        '{}/test.png'.format(self.sample_dir))
            save_images(real_image, image_manifold_size(real_image.shape[0]),
                        '{}/test_real.png'.format(self.sample_dir))

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
            d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            # self.saver = tf.train.Saver()
            could_load, checkpoint_counter = self.load(
                self.checkpoint_dir)

            self.g_sum = merge_summary([self.image_input_sum, self.d__sum,
                                        self.G_sum, self.g_loss_sum])
            self.d_sum = merge_summary(
                [self.image_input_sum, self.image_target_sum, self.d_sum,  self.d_loss_sum])
            self.writer = SummaryWriter(
                os.path.join(globalConfig.p2p_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.p2pGAN')

            nsamples = train_img.shape[0]
            counter = 1
            start_time = time.time()
            batch_idxs = int(nsamples/self.batch_size)
            g_loss = 1
            d_loss = 10
            for epoch in xrange(self.epoch):
                for idx in xrange(0, int(batch_idxs)):
                    batch_target = train_img_rgb[idx *
                                                 self.batch_size:(idx+1)*self.batch_size]
                    batch_labels = train_labels[idx *
                                                self.batch_size:(idx+1)*self.batch_size]
                    batch_input = train_img[idx *
                                            self.batch_size:(idx+1)*self.batch_size]

                    # # add salt & pepper noise
                    # batch_input = noisy("s&p", batch_input)

                    # Update D network
                    if d_loss > 1:
                        _, summary_str, g_loss, d_loss, = self.sess.run([d_optim, self.d_sum, self.g_loss, self.d_loss],
                                                                        feed_dict={
                            self.image_target: batch_target,
                            self.image_input: batch_input,
                            self.label: batch_labels,
                        })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    if g_loss > 0.5:
                        _, summary_str, g_loss, d_loss = self.sess.run([g_optim, self.g_sum, self.g_loss, self.d_loss],
                                                                       feed_dict={
                            self.image_input: batch_input,
                            self.image_target: batch_target,
                            self.label: batch_labels,
                        })
                        self.writer.add_summary(summary_str, counter)

                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, d_loss, g_loss))
                    if np.mod(counter, 1000) == 1:
                        self.save(self.checkpoint_dir, counter)
                # show_all_variables()
                samples, d_loss, g_loss = self.sess.run(
                    [self.sampler, self.d_loss, self.g_loss],
                    feed_dict={
                        self.image_target: test_img_rgb,
                        self.image_input: test_img,
                        self.label: test_labels
                    }
                )
                save_images(samples, image_manifold_size(samples.shape[0]),
                            '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            self.save(self.checkpoint_dir, counter)
    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = "p2pGAN_256.model"
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


def getOneHotedLabel(imgname):
    index = None
    for key, tag in ref.tags.items():
        if tag in imgname:
            index = ref.actions.index(key)
    if index == None:
        index = len(ref.actions)-1
    return ref.oneHoted[index, :]

if __name__ == '__main__':

    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        ds.loadH36M(40960, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=True)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    p2p = P2PGAN()
    p2p.train(ds,val_ds)
    # p2p.test(val_ds)
