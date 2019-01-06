import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')
from data.layers import *
import globalConfig
from data.util import *
from data.dataset import *
import data.ref as ref
Num_of_Joints = ref.nJoints
IMAGE_SIZE_H36M = 128
VALIDATION_SIZE = 5000  # Size of the validation set.
NUM_LABELS = 15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class PoseVAE(object):
    def __init__(
            self, dim_x=Num_of_Joints*3, batch_size=64, lr=1e-3, num_epochs=1000,
            dim_z=46, label_dim=15, n_hidden=40, reuse=False):

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
        self.label_dim = label_dim
        # self.noise_input = tf.placeholder(
        #     dtype=tf.float32, shape=(None, self.dim_z),name='vaenoise')
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_hidden = n_hidden

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

        #latent_variable
        self.z_in = tf.placeholder(
            tf.float32, shape=[None, dim_z], name='latent_variable')

        self.z_in_sum = histogram_summary("z_in", self.z_in)

        self.y, self.z, self.loss, self.neg_marginal_likelihood, self.KL_divergence = self.autoencoder(
            self.x_hat, self.label_hat, self.x, self.dim_x, self.dim_z, self.n_hidden, is_training=True)
        self.y_sum = histogram_summary("y", self.y)
        self.z_sum = histogram_summary("z", self.z)
        self.loss_sum = scalar_summary("loss", self.loss)

        self.neg_marginal_likelihood_sum = scalar_summary(
            "neg_marginal_likelihood", self.neg_marginal_likelihood)
        self.KL_divergence_sum = scalar_summary(
            "KL_divergence", self.KL_divergence)

        t_vars = tf.global_variables()
        self.encoder_vars = [var for var in t_vars if 'encoder' in var.name]
        self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]

        self.saver = tf.train.Saver()

    def sampler(self, z, dim_img, n_hidden):
        y = self.decoder(z, n_hidden, dim_img, is_training=False, reuse=True)
        return y

    def autoencoder(self, x_hat, label_hat, x, dim_img, dim_z, n_hidden, is_training=True, reuse=False):

        mu, sigma = self.encoder(
            x_hat, label_hat, n_hidden, dim_z, reuse=reuse, is_training=is_training)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        # y = self.decoder(z, n_hidden, dim_img,
        #                  is_training=is_training, reuse=reuse)
        # y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        y = self.decoder(z, n_hidden, dim_img,
                                    is_training=False, reuse=reuse)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.square(y-x)
        marginal_likelihood = 0.5*tf.reduce_sum(marginal_likelihood)*100
        # marginal_likelihood = tf.reduce_sum(
        #     x * tf.log(y) + (1 - x) * tf.log(1 - y), [1])
        # marginal_likelihood = -tf.reduce_mean(marginal_likelihood)

        KL_divergence = 0.5 * \
            tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                          tf.log(1e-8 + tf.square(sigma)) - 1, [1])

        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -marginal_likelihood - KL_divergence

        loss = -ELBO

        return y, z, loss, marginal_likelihood, KL_divergence

    def encoder(self, x, label, n_hidden, n_output, is_training=True, reuse=True):
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()

            x = concat([x, label], 1)
            # 1st hidden layer
            h0 = lrelu(
                bn(dropout(linear(x, n_hidden, 'h0_lin'),
                           is_training=is_training), is_training=is_training, scope="h0_bn"))

            h0 = concat([h0, label], 1)

            # 2nd hidden layer
            h1 = lrelu(
                bn(dropout(linear(h0, n_hidden, 'h1_lin'),
                           is_training=is_training), is_training=is_training, scope="h1_bn"))

            h1 = concat([h1, label], 1)

            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
            # wo = tf.get_variable(
            #     'wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
            # bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            # gaussian_params = tf.matmul(h1, wo) + bo

            gaussian_params = linear(h1, n_output*2)

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :n_output]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
            return mean, stddev

    def decoder(self, z, n_hidden, n_output, is_training=True, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):

            # 1st hidden layer
            h0 = lrelu(
                bn(dropout(linear(z, n_hidden, 'h0_lin'),
                           is_training=is_training), is_training=is_training, scope="h0_bn"))

            # 2nd hidden layer
            h1 = lrelu(
                bn(dropout(linear(h0, n_hidden, 'h1_lin'),
                           is_training=is_training), is_training=is_training, scope="h1_bn"))

            # output layer-mean
            # y = tf.sigmoid(tf.matmul(h1, wo) + bo)
            y = tf.tanh(
                bn(dropout(linear(h1, n_output, 'y_lin'),
                           is_training=is_training), is_training=is_training, scope="y_bn"))
        return y

    def test(self, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        test_labels, test_skel, _, _, n_samples, total_batch = prep_data(
            valid_dataset, self.batch_size)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()
            counter = 1
            could_load, checkpoint_counter = self.load(
                self.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            for epoch in range(total_batch):
                offset = (epoch * self.batch_size) % (n_samples)
                batch_xs_input = test_skel[
                    offset:(offset + self.batch_size), :]

                batch_label_input = test_labels[offset:(
                    offset + self.batch_size), :]

                samples = self.sess.run(
                    self.y, feed_dict={self.x_hat: batch_xs_input, self.label_hat: batch_label_input})

                images=np.ones((samples.shape[0], 128, 128, 3))
                save_images(images, image_manifold_size(samples.shape[0]),
                            '{}/test_{:02d}.png'.format(self.sample_dir, epoch), skel=samples)
                            
    def predict(self, valid_dataset, amplify_ratio=9):
        test_labels, test_skel, _, _, _, _, = prep_data(
            valid_dataset, self.batch_size)
        n_samples = test_skel.shape[0]

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()
            counter = 1
            could_load, checkpoint_counter = self.load(
                self.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            samples_label = np.zeros((
                n_samples*amplify_ratio, test_labels.shape[1]))
            samples_skel = np.zeros((
                n_samples*amplify_ratio, test_skel.shape[1]))

            for i in range(amplify_ratio):
                for idx in range(n_samples//self.batch_size):
                    offset = idx*self.batch_size
                    start_idx = offset
                    stop_idx = min(offset+self.batch_size, n_samples)
                    batch_skel = test_skel[start_idx:stop_idx]
                    batch_label = test_labels[start_idx:stop_idx]
                    predicted = ((self.sess.run(
                        self.y, feed_dict={self.x_hat: batch_skel, self.label_hat: batch_label}))*256-128)
                    samples_skel[i*n_samples+start_idx:i *
                                 n_samples+stop_idx, :] = predicted
                    samples_label[i*n_samples+start_idx:i *
                                  n_samples+stop_idx, :] = batch_label

            samples_label = np.asarray(samples_label, dtype=np.uint8)
            samples_skel = np.asarray(samples_skel, dtype=np.float32)

            np.save('samples_skel.out', samples_skel)
            np.save('samples_label.out', samples_label)


    def train(self, train_dataset, valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        train_labels, train_skel, _, _, n_samples, total_batch = prep_data(
            train_dataset, self.batch_size)
        test_labels, test_skel, _, _, _, _ = prep_data(
            valid_dataset, self.batch_size)

        min_tot_loss = 1e6

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.Session() as self.sess:
            counter = 1
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
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

                    batch_label_input = train_labels[offset:(
                        offset + self.batch_size), :]
                    batch_xs_target = batch_xs_input

                    _, tot_loss, loss_likelihood, loss_divergence, summary_str = self.sess.run(
                        (self.train_op, self.loss,
                         self.neg_marginal_likelihood, self.KL_divergence, self.sum),
                        feed_dict={self.x_hat: batch_xs_input, self.label_hat: batch_label_input,
                                   self.x: batch_xs_target})
                    self.writer.add_summary(summary_str, counter)
                    # print cost every epoch
                    counter += 1
                print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                    epoch, tot_loss, loss_likelihood, loss_divergence))

                # if minimum loss is updated or final epoch, plot results
                if min_tot_loss > tot_loss or epoch+1 == self.num_epochs or np.mod(counter, 2000) == 1:
                    min_tot_loss = tot_loss
                    samples = self.sess.run(
                        self.y, feed_dict={self.x_hat: test_skel, self.label_hat: test_labels})

                    images=np.ones((samples.shape[0], 128, 128, 3))
                    save_images(images, image_manifold_size(samples.shape[0]),
                                '{}/train_{:02d}.png'.format(self.sample_dir, epoch), skel=samples)

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
        ds = Dataset()
        ds.loadH36M(64*50, mode='train', tApp=True, replace=False)

        # val_ds = Dataset()
        # val_ds.loadH36M(64, mode='valid', tApp=True, replace=False)
    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(10240, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadApe(2048, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    vae = PoseVAE()
    # vae.train(ds, val_ds)
    # vae.test(val_ds)
    vae.predict(ds)
