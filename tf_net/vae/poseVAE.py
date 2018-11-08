import sys
sys.path.append('./')
import data.ref as ref
from data.dataset import *
import globalConfig
import plot_utils
import os
import numpy as np
import tensorflow as tf



IMAGE_SIZE_H36M = 128
Num_of_Joints = ref.nJoints
VALIDATION_SIZE = 5000  # Size of the validation set.
NUM_LABELS = 15


class PoseVAE(object):
    def __init__(
            self, dim_x=Num_of_Joints*3, batch_size=128, lr=1e-3, num_epochs=250,
            b1=0.5, dim_z=20, n_hidden=20, ADD_NOISE=True, PRR=True, PRR_n_img_x=10, PRR_n_img_y=10, PRR_resize_factor=1.0,
            PMLR=False, PMLR_n_img_x=20, PMLR_n_img_y=20, PMLR_resize_factor=1.0, PMLR_z_range=2.0, PMLR_n_samples=5000):
        #dim_z=dim of noise
        self.dim_z = dim_z
        #dim_x: dim of input
        self.dim_x = dim_x
        self.ADD_NOISE = ADD_NOISE
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.num_epochs = num_epochs
        self.RESULTS_DIR = './vae/result'
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

        self.x_hat = tf.placeholder(
            tf.float32, shape=[None, dim_x], name='input_pose')
        self.x = tf.placeholder(
            tf.float32, shape=[None, dim_x], name='target_pose')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.z_in = tf.placeholder(
            tf.float32, shape=[None, dim_z], name='latent_variable')

        self.y, self.z, self.loss, self.neg_marginal_likelihood, self.KL_divergence = self.autoencoder(
            self.x_hat, self.x, self.dim_x, self.dim_z, self.n_hidden, self.keep_prob)

        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def decoder(self, z, dim_img, n_hidden):
        y = self.bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
        return y

    def autoencoder(self, x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
        mu, sigma = self.gaussian_MLP_encoder(
            x_hat, n_hidden, dim_z, keep_prob)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self.bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(
            x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
        KL_divergence = 0.5 * \
            tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                          tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence

        loss = -ELBO

        return y, z, loss, -marginal_likelihood, KL_divergence

    def gaussian_MLP_encoder(self, x, n_hidden, n_output, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable(
                'w0', [x.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(x, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable(
                'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, keep_prob)

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

    def bernoulli_MLP_decoder(self, z, n_hidden, n_output, keep_prob, reuse=False):
        with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable(
                'w0', [z.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable(
                'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, keep_prob)

            # output layer-mean
            wo = tf.get_variable(
                'wo', [h1.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            y = tf.sigmoid(tf.matmul(h1, wo) + bo)

        return y

    def train(self, train_dataset, valid_dataset):
        train_size = len(train_dataset.frmList)
        train_data = []
        train_labels = []
        for frm in train_dataset.frmList:
            train_data.append(frm.skel)
            train_labels.append(frm.label)

        test_data = []
        test_labels = []
        for frm in valid_dataset.frmList:
            test_data.append(frm.skel)
            test_labels.append(frm.label)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        train_data = train_data/max(-1*train_data.min(), train_data.max())
        test_data = test_data/max(-1*test_data.min(), test_data.max())

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, :]
        validation_labels = train_labels[:VALIDATION_SIZE, :]
        train_data = train_data[VALIDATION_SIZE:, :]
        train_labels = train_labels[VALIDATION_SIZE:, :]

        train_total_data = np.concatenate(
            (train_data, train_labels), axis=1)

        train_size = train_total_data.shape[0]

        n_samples = train_size
        batch_size = self.batch_size
        total_batch = int(n_samples / self.batch_size)
        min_tot_loss = 1e99

        if self.PRR:
            PRR = plot_utils.Plot_Reproduce_Performance(
                self.RESULTS_DIR, self.PRR_n_img_x, self.PRR_n_img_y, IMAGE_SIZE_H36M, IMAGE_SIZE_H36M,
                self.PRR_resize_factor)

            x_PRR = test_data[0:PRR.n_tot_imgs, :]

            x_PRR_img = x_PRR.reshape(
                PRR.n_tot_imgs, -1)

            PRR.save_images(x_PRR_img, name='input.jpg')

            if self.ADD_NOISE:
                x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
                x_PRR += np.random.randint(2, size=x_PRR.shape)

                x_PRR_img = x_PRR.reshape(
                    PRR.n_tot_imgs, -1)
                PRR.save_images(x_PRR_img, name='input_noise.jpg')

        # Plot for manifold learning result
        if self.PMLR and self.dim_z == 2:

            PMLR = plot_utils.Plot_Manifold_Learning_Result(
                self.RESULTS_DIR, self.PMLR_n_img_x, self.PMLR_n_img_y, IMAGE_SIZE_H36M, IMAGE_SIZE_H36M, self.PMLR_resize_factor, self.PMLR_z_range)

            x_PMLR = test_data[0:self.PMLR_n_samples, :]
            id_PMLR = test_labels[0:self.PMLR_n_samples, :]

            if self.ADD_NOISE:
                x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
                x_PMLR += np.random.randint(2, size=x_PMLR.shape)

            decoded = vae.decoder(z_in, dim_img, n_hidden)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer(),
                     feed_dict={self.keep_prob: 0.9})

            for epoch in range(self.num_epochs):

                # Random shuffling
                np.random.shuffle(train_total_data)
                train_data_ = train_total_data[:, :-NUM_LABELS]

                # Loop over all batches
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % (n_samples)
                    batch_xs_input = train_data_[
                        offset:(offset + batch_size), :]

                    batch_xs_target = batch_xs_input

                    # add salt & pepper noise
                    if self.ADD_NOISE:
                        batch_xs_input = batch_xs_input * \
                            np.random.randint(2, size=batch_xs_input.shape)
                        batch_xs_input += np.random.randint(
                            2, size=batch_xs_input.shape)

                    _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                        (self.train_op, self.loss,
                         self.neg_marginal_likelihood, self.KL_divergence),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 0.9})

                # print cost every epoch
                print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                    epoch, tot_loss, loss_likelihood, loss_divergence))

                # if minimum loss is updated or final epoch, plot results
                if min_tot_loss > tot_loss or epoch+1 == self.num_epochs:
                    min_tot_loss = tot_loss
                    # Plot for reproduce performance
                    if self.PRR:
                        y_PRR = sess.run(
                            self.y, feed_dict={self.x_hat: x_PRR, self.keep_prob: 1})
                        y_PRR_img = y_PRR.reshape(
                            PRR.n_tot_imgs, -1)
                        PRR.save_images(
                            y_PRR_img, name="/PRR_epoch_%02d" % (epoch) + ".jpg")

                    # Plot for manifold learning result
                    if self.PMLR and self.dim_z == 2:
                        y_PMLR = sess.run(decoded, feed_dict={
                            self.z_in: self.PMLR.z, self.keep_prob: 1})
                        y_PMLR_img = y_PMLR.reshape(
                            PMLR.n_tot_imgs, -1)
                        self.PMLR.save_images(
                            y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

                        # plot distribution of labeled images
                        z_PMLR = sess.run(
                            z, feed_dict={self.x_hat: self.x_PMLR, self.keep_prob: 1})
                        self.PMLR.save_scattered_image(
                            z_PMLR, self.id_PMLR, name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")


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

    vae = PoseVAE()
    vae.train(ds, val_ds)

    # train_total_data, train_size=ds
    # , _, _, test_data, test_labels = ds
