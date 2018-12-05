import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('./')
import data.ref as ref
from data.dataset import *
from data.util import *
import data.plot_utils as plot_utils
import globalConfig
from data.layers import *
IMAGE_SIZE_H36M = 128
Num_of_Joints = ref.nJoints
VALIDATION_SIZE = 5000  # Size of the validation set.
NUM_LABELS = 15


class PoseVAE(object):
    def __init__(
            self, dim_x=Num_of_Joints*3, batch_size=64, lr=1e-3, num_epochs=3000,
            dim_z=46, label_dim=15, n_hidden=40, PRR=True, PRR_n_img_x=8, PRR_n_img_y=8, PRR_resize_factor=1.0,
            PMLR=True, PMLR_n_img_x=20, PMLR_n_img_y=20, PMLR_resize_factor=1.0, PMLR_z_range=2.0, PMLR_n_samples=5000, reuse=False):

        checkpoint_dir = 'checkpoint'
        self.checkpoint_dir = os.path.join(
            globalConfig.vae_pretrain_path, checkpoint_dir)
        sample_dir = 'samples'
        self.sample_dir = os.path.join(
            globalConfig.vae_pretrain_path, sample_dir)
        #dim_z=dim of latent space
        self.dim_z = dim_z
        #dim_x: dim of input pose dimension
        self.dim_x = dim_x
        self.label_dim=label_dim
        # self.noise_input = tf.placeholder(
        #     dtype=tf.float32, shape=(None, self.dim_z),name='vaenoise')
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_hidden = n_hidden
        self.PRR = PRR
        self.PRR_n_img_x = PRR_n_img_x
        self.PRR_n_img_y = PRR_n_img_y
        self.PRR_resize_factor = PRR_resize_factor
        self.PMLR = PMLR
        self.PMLR_n_img_x = PMLR_n_img_x
        self.PMLR_n_img_y = PMLR_n_img_y
        self.PMLR_resize_factor = PMLR_resize_factor
        self.PMLR_z_range = PMLR_z_range
        self.PMLR_n_samples = PMLR_n_samples
        #input_pose
        self.x_hat = tf.placeholder(
            tf.float32, shape=[None, dim_x], name='input_pose')
        self.label_hat = tf.placeholder(
            tf.float32, shape=[None, self.label_dim], name='input_label')
        #target_pose
        self.x = tf.placeholder(
            tf.float32, shape=[None, dim_x], name='target_pose')

        self.x_hat_sum = histogram_summary("x_hat", self.x_hat)
        self.label_hat_sum = histogram_summary("label_hat", self.label_hat)
        self.x_sum = histogram_summary("x", self.x)
        

        self.keep_prob = 0.5
        #latent_variable
        self.z_in = tf.placeholder(
            tf.float32, shape=[None, dim_z], name='latent_variable')
        self.z_in_sum = histogram_summary("z_in", self.z_in)

        self.y, self.z, self.loss, self.neg_marginal_likelihood, self.KL_divergence = self.autoencoder(
            self.x_hat, self.label_hat, self.x, self.dim_x, self.dim_z, self.n_hidden, self.keep_prob)
        self.y_sum = histogram_summary("y", self.y)
        self.z_sum = histogram_summary("z", self.z)
        self.loss_sum = scalar_summary("loss", self.loss)

        self.neg_marginal_likelihood_sum = scalar_summary(
            "neg_marginal_likelihood", self.neg_marginal_likelihood)
        self.KL_divergence_sum = scalar_summary(
            "KL_divergence", self.KL_divergence)


        t_vars = tf.trainable_variables()
        self.encoder_vars = [var for var in t_vars if 'encoder' in var.name]
        self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]


        self.saver = tf.train.Saver()

    def sampler(self, z, dim_img, n_hidden):
        y = self.decoder(z, n_hidden, dim_img, 1.0, reuse=True)
        return y

    def autoencoder(self, x_hat, label_hat, x, dim_img, dim_z, n_hidden, keep_prob, reuse=False):

        mu, sigma = self.encoder(
            x_hat, label_hat, n_hidden, dim_z, keep_prob, reuse=reuse)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self.decoder(z, n_hidden, dim_img, keep_prob)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(
            x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
        marginal_likelihood = tf.square(y-x)
        marginal_likelihood = tf.reduce_sum(marginal_likelihood/2)


        KL_divergence = 0.5 * \
            tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                          tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood + KL_divergence

        loss = ELBO

        return y, z, loss, marginal_likelihood, KL_divergence

    def encoder(self, x,label, n_hidden, n_output, keep_prob, reuse=True):
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            x = concat([x, label], 1)
            # 1st hidden layer
            w0 = tf.get_variable(
                'w0', [x.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(x, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, keep_prob)
            h0=concat([h0, label], 1)

            # 2nd hidden layer
            w1 = tf.get_variable(
                'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, keep_prob)
            h1=concat([h1, label], 1)

            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
            wo = tf.get_variable(
                'wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
            bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            gaussian_params = tf.matmul(h1, wo) + bo

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :n_output]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
            return mean, stddev

    def decoder(self, z, n_hidden, n_output, keep_prob, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable(
                'w0', [z.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.leaky_relu(h0)
            h0 = tf.nn.dropout(h0, keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable(
                'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            # h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, keep_prob)

            # output layer-mean
            wo = tf.get_variable(
                'wo', [h1.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            y = tf.matmul(h1, wo) + bo
            # y = tf.sigmoid(tf.matmul(h1, wo) + bo)

        return y

    def train(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        show_all_variables()

        train_labels, train_skel, _, test_labels, test_skel, _, n_samples, total_batch = prep_data(
            train_dataset, valid_dataset, self.batch_size)

        min_tot_loss = 1e6

        if self.PRR:
            PRR = plot_utils.Plot_Reproduce_Performance(
                self.sample_dir, self.PRR_n_img_x, self.PRR_n_img_y, IMAGE_SIZE_H36M, IMAGE_SIZE_H36M,
                self.PRR_resize_factor)

            x_PRR = test_skel[0:PRR.n_tot_imgs, :]
            lable_PRR = test_labels[0:PRR.n_tot_imgs, :]
            x_PRR_img = x_PRR.reshape(
                PRR.n_tot_imgs, -1)

            PRR.save_images(x_PRR_img, name='input.jpg')

            # x_PRR = x_PRR * np.random.normal(0, 0.05, size=x_PRR.shape)
            # x_PRR += np.random.randint(2, size=x_PRR.shape)

            x_PRR_img = x_PRR.reshape(
                PRR.n_tot_imgs, -1)
            PRR.save_images(x_PRR_img, name='input_noise.jpg')

        # Plot for manifold learning result
        if self.PMLR and self.dim_z == 2:

            PMLR = plot_utils.Plot_Manifold_Learning_Result(
                self.sample_dir, self.PMLR_n_img_x, self.PMLR_n_img_y, IMAGE_SIZE_H36M, IMAGE_SIZE_H36M, self.PMLR_resize_factor, self.PMLR_z_range)

            x_PMLR = test_skel[0:self.PMLR_n_samples, :]
            id_PMLR = test_labels[0:self.PMLR_n_samples, :]

            # x_PMLR = x_PMLR * np.random.normal(0, 0.05, size=x_PMLR.shape)
            # x_PMLR += np.random.randint(2, size=x_PMLR.shape)

            decoded = vae.sampler(self.z_in, self.dim_x, self.n_hidden)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
            self.loss)
        with tf.Session() as self.sess:
            counter = 1
            # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            could_load = False
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            tf.global_variables_initializer().run()
            self.sum = merge_summary([self.x_sum, self.x_hat_sum, self.label_hat_sum, self.z_sum, self.y_sum,
                                      self.neg_marginal_likelihood_sum, self.KL_divergence_sum, self.loss_sum])
            self.writer = SummaryWriter(
                os.path.join(globalConfig.vae_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.poseVAE')


            for epoch in range(self.num_epochs):

                # Loop over all batches
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * self.batch_size) % (n_samples)
                    batch_xs_input = train_skel[
                        offset:(offset + self.batch_size), :]

                    batch_label_input = train_labels[
                        offset:(offset + self.batch_size), :]
                    batch_xs_target = batch_xs_input
                    # add salt & pepper noise
                    # batch_xs_input = batch_xs_input * \
                    #     np.random.normal(0, 0.05, batch_xs_input.shape)
                    # np.random.randint(
                    #     0.9, high=0.1, size=batch_xs_input.shape)
                    # batch_xs_input += np.random.randint(
                    #     2, size=batch_xs_input.shape)

                    _, tot_loss, loss_likelihood, loss_divergence, summary_str = self.sess.run(
                        (self.train_op, self.loss,
                         self.neg_marginal_likelihood, self.KL_divergence, self.sum),
                        feed_dict={self.x_hat: batch_xs_input, self.label_hat: batch_label_input, self.x: batch_xs_target})

                    self.writer.add_summary(summary_str, counter)
                    # print cost every epoch
                    counter += 1
                print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                    epoch, tot_loss, loss_likelihood, loss_divergence))

                # if minimum loss is updated or final epoch, plot results
                if min_tot_loss > tot_loss or epoch+1 == self.num_epochs or np.mod(counter, 2000) == 1:
                    min_tot_loss = tot_loss
                    # Plot for reproduce performance
                    if self.PRR:
                        y_PRR = self.sess.run(
                            self.y, feed_dict={self.x_hat: x_PRR, self.label_hat: lable_PRR})
                        y_PRR_img = y_PRR.reshape(
                            PRR.n_tot_imgs, -1)
                        PRR.save_images(
                            y_PRR_img, name="/PRR_epoch_%02d" % (epoch) + ".jpg")

                    # Plot for manifold learning result
                    if self.PMLR and self.dim_z == 2:
                        y_PMLR = self.sess.run(decoded, feed_dict={
                            self.z_in: PMLR.z})
                        y_PMLR_img = y_PMLR.reshape(
                            PMLR.n_tot_imgs, -1)
                        PMLR.save_images(
                            y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

                        # plot distribution of labeled images
                        z_PMLR = self.sess.run(
                            self.z, feed_dict={self.x_hat: x_PMLR})
                        PMLR.save_scattered_image(
                            z_PMLR, id_PMLR, name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")

                    if min_tot_loss > tot_loss or epoch+1 == self.num_epochs:
                        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}".format(
            globalConfig.dataset, self.batch_size)

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
        model_name = "POSEVAE.model"
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
        # for i in range(0, 20000, 20000):
        ds.loadH36M(40960, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        # for i in range(0, 20000, 20000):
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    vae = PoseVAE()
    vae.train(ds, val_ds)
