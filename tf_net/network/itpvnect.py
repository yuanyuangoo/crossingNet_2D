import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import sys
import os
sys.path.append('./')
from data.util import *
from data.dataset import *
import globalConfig
from six.moves import xrange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class vnect():
    def __init__(self, input_size=128, checkpoint_dir="./checkpoint", sample_dir="samples", batch_size=64, learning_rate=2e-5, beta1=0.5, epoch=200, conv_ratio=8):
        self.is_training = False
        self.dataset_name=globalConfig.dataset
        self.checkpoint_dir = os.path.join(
            globalConfig.vnect_pretrain_path, checkpoint_dir)
        self.input_size = input_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.sample_dir = os.path.join(
            globalConfig.vnect_pretrain_path, sample_dir)
        self.conv_ratio = conv_ratio
        self.image_input = tf.placeholder(dtype=tf.float32,
                                          shape=(None, input_size, input_size, 3))
        self.image_input_sum = image_summary("image_input", self.image_input)
        self.n_joints = ref.nJoints
        self.pose_input_heat_map = tf.placeholder(
            dtype=tf.float32, shape=(None, input_size//8, input_size//8, self.n_joints))
        self.z_heatmap_gt = tf.placeholder(dtype=tf.float32, shape=(
            None, input_size//8, input_size//8, self.n_joints))

        self._create_network()

        self.heatmap_loss = tf.nn.l2_loss(
            self.heatmap - self.pose_input_heat_map, name='heatmap_loss')+tf.nn.l2_loss(
            self.heatmap_intermidiate1 - self.pose_input_heat_map, name='heatmap_intermidiate1_loss')+tf.nn.l2_loss(
            self.heatmap_intermidiate2 - self.pose_input_heat_map, name='heatmap_intermidiate2_loss')

        self.z_loss = tf.nn.l2_loss(
            tf.multiply(self.z_heatmap - self.z_heatmap_gt, self.pose_input_heat_map), name='z_loss')+tf.nn.l2_loss(
            tf.multiply(self.z_heatmap_intermidiate1 - self.z_heatmap_gt, self.pose_input_heat_map), name='z_intermidiate1_loss')+tf.nn.l2_loss(
            tf.multiply(self.z_heatmap_intermidiate2 - self.z_heatmap_gt, self.pose_input_heat_map), name='z_intermidiate2_loss')

        self.loss = self.heatmap_loss+self.z_loss
        self.loss_sum = scalar_summary('loss', self.loss)
        self.heatmap_loss_sum = scalar_summary(
            'heatmap_loss', self.heatmap_loss)
        self.z_loss_sum = scalar_summary('z_loss', self.z_loss)
        self.t_vars = tf.global_variables()
        self.saver = tf.train.Saver(max_to_keep=20)

    def predict(self, valid_dataset, train_total_batch):
        self.total_batch = train_total_batch
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        _, test_skel, _, test_img_rgb, test_heat_maps, test_z_heat_maps, _, _ = prep_data(
            valid_dataset, self.batch_size, heat_map=True)
        result = np.zeros(test_skel.shape)
        with tf.Session() as self.sess:
            counter = 1
            start_time = time.time()
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            print(counter)
            batch_idxs = test_skel.shape[0]//self.batch_size

            for idx in xrange(0, int(batch_idxs)):
                heatmap_samples, z_heatmap_samples = self.sess.run([self.heatmap, self.z_heatmap],
                                                                   feed_dict={
                    self.image_input: test_img_rgb[idx *
                                                   self.batch_size:(idx+1)*self.batch_size]
                })
                skel = SkelFromHeatmap(heatmap_samples, z_heatmap_samples)
                result[idx * self.batch_size:(idx+1)*self.batch_size] = skel
                save_images(test_img_rgb, image_manifold_size(self.batch_size),
                            '{}/test_{:02d}.png'.format(self.sample_dir, idx), skel=skel)
            a = eval_pck(result, test_skel,1,1,1)
            

    def train(self,train_dataset,valid_dataset):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
            
        _, train_skel, _, train_img_rgb, train_heat_maps, train_z_heat_maps, self.n_samples, self.total_batch = prep_data(
            train_dataset, self.batch_size, heat_map=True)
        _, test_skel, _, test_img_rgb, test_heat_maps, test_z_heat_maps, _, _ = prep_data(
            valid_dataset, self.batch_size, heat_map=True)

        with tf.Session() as self.sess:
            a_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss, var_list=self.t_vars)
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            self.h_sum = merge_summary(
                [self.loss_sum, self.heatmap_loss_sum, self.image_input_sum])
            self.all_sum = merge_summary([
                                          self.z_loss_sum, self.loss_sum, self.heatmap_loss_sum, self.image_input_sum])
            self.writer = SummaryWriter(
                os.path.join(globalConfig.vnect_pretrain_path, "logs"), graph=self.sess.graph, filename_suffix='.ivtvnect')

            counter = 0
            start_time = time.time()

            # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            # if could_load:
            #     counter = checkpoint_counter
            #     print(" [*] Load SUCCESS")
            # else:
            #     print(" [!] Load failed...")

            self.train_size = np.inf
            for epoch in xrange(self.epoch):
                batch_idxs = min(
                    len(train_img_rgb), self.train_size) // self.batch_size

                for idx in xrange(0, int(batch_idxs)):
                    batch_images = train_img_rgb[idx *
                                                 self.batch_size:(idx+1)*self.batch_size]
                    batch_heatmaps = train_heat_maps[idx *
                                                     self.batch_size:(idx+1)*self.batch_size]
                    batch_z_heatmaps = train_z_heat_maps[idx *
                                                         self.batch_size:(idx+1)*self.batch_size]
                    # Update network
                    _, summary_str, heatmap_loss, z_loss, loss = self.sess.run([a_optim, self.all_sum, self.heatmap_loss, self.z_loss, self.loss],
                                                                               feed_dict={
                        self.image_input: batch_images,
                        self.pose_input_heat_map: batch_heatmaps,
                        self.z_heatmap_gt: batch_z_heatmaps
                    })
                    self.writer.add_summary(summary_str, counter)
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, heatmap_loss: %.8f, z_loss: %.8f, loss: %.8f "
                          % (epoch, self.epoch, idx, batch_idxs,
                             time.time() - start_time, heatmap_loss, z_loss, loss))
                    counter += 1

                heatmap_samples, z_heatmap_samples = self.sess.run([self.heatmap, self.z_heatmap],
                                                                   feed_dict={
                    self.image_input: test_img_rgb
                })

                skel=SkelFromHeatmap(heatmap_samples, z_heatmap_samples)
                
                save_images(test_img_rgb, image_manifold_size(self.batch_size),
                            '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx), skel=skel)
                if np.mod(epoch,50)==0:
                    self.save(self.checkpoint_dir, counter)
            self.save(self.checkpoint_dir, counter)

    def _create_network(self):
        # Conv
        self.conv1 = tc.layers.conv2d(
            self.image_input, kernel_size=7, num_outputs=64, stride=2, scope='conv1')
        self.pool1 = tc.layers.max_pool2d(
            self.conv1, kernel_size=3, padding='same', scope='pool1')

        # Residual block 2a
        self.res2a_branch2a = tc.layers.conv2d(
            self.pool1, kernel_size=1, num_outputs=64, scope='res2a_branch2a')
        self.res2a_branch2b = tc.layers.conv2d(
            self.res2a_branch2a, kernel_size=3, num_outputs=64, scope='res2a_branch2b')
        self.res2a_branch2c = tc.layers.conv2d(
            self.res2a_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch2c')
        self.res2a_branch1 = tc.layers.conv2d(
            self.pool1, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch1')
        self.res2a = tf.add(self.res2a_branch2c,
                            self.res2a_branch1, name='res2a_add')
        self.res2a = tf.nn.relu(self.res2a, name='res2a')

        # Residual block 2b
        self.res2b_branch2a = tc.layers.conv2d(
            self.res2a, kernel_size=1, num_outputs=64, scope='res2b_branch2a')
        self.res2b_branch2b = tc.layers.conv2d(
            self.res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2b_branch2b')
        self.res2b_branch2c = tc.layers.conv2d(
            self.res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2b_branch2c')
        self.res2b = tf.add(self.res2b_branch2c, self.res2a, name='res2b_add')
        self.res2b = tf.nn.relu(self.res2b, name='res2b')

        # Residual block 2c
        self.res2c_branch2a = tc.layers.conv2d(
            self.res2b, kernel_size=1, num_outputs=64, scope='res2c_branch2a')
        self.res2c_branch2b = tc.layers.conv2d(
            self.res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2c_branch2b')
        self.res2c_branch2c = tc.layers.conv2d(
            self.res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2c_branch2c')
        self.res2c = tf.add(self.res2c_branch2c, self.res2b, name='res2c_add')
        self.res2c = tf.nn.relu(self.res2b, name='res2c')

        # Residual block 3a
        self.res3a_branch2a = tc.layers.conv2d(
            self.res2c, kernel_size=1, num_outputs=128, stride=2, scope='res3a_branch2a')
        self.res3a_branch2b = tc.layers.conv2d(
            self.res3a_branch2a, kernel_size=3, num_outputs=128, scope='res3a_branch2b')
        self.res3a_branch2c = tc.layers.conv2d(
            self.res3a_branch2b, kernel_size=1, num_outputs=512, activation_fn=None, scope='res3a_branch2c')
        self.res3a_branch1 = tc.layers.conv2d(
            self.res2c, kernel_size=1, num_outputs=512, activation_fn=None, stride=2, scope='res3a_branch1')
        self.res3a = tf.add(self.res3a_branch2c,
                            self.res3a_branch1, name='res3a_add')
        self.res3a = tf.nn.relu(self.res3a, name='res3a')

        # Residual block 3b
        self.res3b_branch2a = tc.layers.conv2d(
            self.res3a, kernel_size=1, num_outputs=128, scope='res3b_branch2a')
        self.res3b_branch2b = tc.layers.conv2d(
            self.res3b_branch2a, kernel_size=3, num_outputs=128, scope='res3b_branch2b')
        self.res3b_branch2c = tc.layers.conv2d(
            self.res3b_branch2b, kernel_size=1, num_outputs=512, activation_fn=None, scope='res3b_branch2c')
        self.res3b = tf.add(self.res3b_branch2c, self.res3a, name='res3b_add')
        self.res3b = tf.nn.relu(self.res3b, name='res3b')

        # Residual block 3c
        self.res3c_branch2a = tc.layers.conv2d(
            self.res3b, kernel_size=1, num_outputs=128, scope='res3c_branch2a')
        self.res3c_branch2b = tc.layers.conv2d(
            self.res3c_branch2a, kernel_size=3, num_outputs=128, scope='res3c_branch2b')
        self.res3c_branch2c = tc.layers.conv2d(
            self.res3c_branch2b, kernel_size=1, num_outputs=512, activation_fn=None, scope='res3c_branch2c')
        self.res3c = tf.add(self.res3c_branch2c, self.res3b, name='res3c_add')
        self.res3c = tf.nn.relu(self.res3c, name='res3c')

        # Residual block 3d
        self.res3d_branch2a = tc.layers.conv2d(
            self.res3c, kernel_size=1, num_outputs=128, scope='res3d_branch2a')
        self.res3d_branch2b = tc.layers.conv2d(
            self.res3d_branch2a, kernel_size=3, num_outputs=128, scope='res3d_branch2b')
        self.res3d_branch2c = tc.layers.conv2d(
            self.res3d_branch2b, kernel_size=1, num_outputs=512, activation_fn=None, scope='res3d_branch2c')
        self.res3d = tf.add(self.res3d_branch2c, self.res3b, name='res3d_add')
        self.res3d = tf.nn.relu(self.res3d, name='res3d')

        # Residual block 4a
        self.res4a_branch2a = tc.layers.conv2d(
            self.res3d, kernel_size=1, num_outputs=256, stride=2, scope='res4a_branch2a')
        self.res4a_branch2b = tc.layers.conv2d(
            self.res4a_branch2a, kernel_size=3, num_outputs=256, scope='res4a_branch2b')
        self.res4a_branch2c = tc.layers.conv2d(
            self.res4a_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4a_branch2c')
        self.res4a_branch1 = tc.layers.conv2d(
            self.res3d, kernel_size=1, num_outputs=1024, activation_fn=None, stride=2, scope='res4a_branch1')
        self.res4a = tf.add(self.res4a_branch2c,
                            self.res4a_branch1, name='res4a_add')
        self.res4a = tf.nn.relu(self.res4a, name='res4a')

        # Residual block 4b
        self.res4b_branch2a = tc.layers.conv2d(
            self.res4a, kernel_size=1, num_outputs=256, scope='res4b_branch2a')
        self.res4b_branch2b = tc.layers.conv2d(
            self.res4b_branch2a, kernel_size=3, num_outputs=256, scope='res4b_branch2b')
        self.res4b_branch2c = tc.layers.conv2d(
            self.res4b_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4b_branch2c')
        self.res4b = tf.add(self.res4b_branch2c, self.res4a, name='res4b_add')
        self.res4b = tf.nn.relu(self.res4b, name='res4b')

        # Residual block 4c
        self.res4c_branch2a = tc.layers.conv2d(
            self.res4b, kernel_size=1, num_outputs=256, scope='res4c_branch2a')
        self.res4c_branch2b = tc.layers.conv2d(
            self.res4c_branch2a, kernel_size=3, num_outputs=256, scope='res4c_branch2b')
        self.res4c_branch2c = tc.layers.conv2d(
            self.res4c_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4c_branch2c')
        self.res4c = tf.add(self.res4c_branch2c, self.res4b, name='res4c_add')
        self.res4c = tf.nn.relu(self.res4c, name='res4c')

        # Residual block 4d
        self.res4d_branch2a = tc.layers.conv2d(
            self.res4c, kernel_size=1, num_outputs=256, scope='res4d_branch2a')
        self.res4d_branch2b = tc.layers.conv2d(
            self.res4d_branch2a, kernel_size=3, num_outputs=256, scope='res4d_branch2b')
        self.res4d_branch2c = tc.layers.conv2d(
            self.res4d_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4d_branch2c')
        self.res4d = tf.add(self.res4d_branch2c, self.res4c, name='res4d_add')
        self.res4d = tf.nn.relu(self.res4d, name='res4d')
        self.res4d_heatmap1a = tf.layers.conv2d_transpose(
            self.res4d, kernel_size=4, filters=self.n_joints*2, activation=None, strides=2, padding='same', use_bias=False, name='res4f_heatmap1a')
        self.res4d_heatmap_bn = tc.layers.batch_norm(
            self.res4d_heatmap1a, scale=True, is_training=self.is_training, scope='res4d_heatmap_bn')
        self.heatmap_intermidiate1, self.z_heatmap_intermidiate1 = tf.split(tf.nn.relu(
            self.res4d_heatmap_bn, name='intermidiate1'), num_or_size_splits=2, axis=3)

        # Residual block 4e
        self.res4e_branch2a = tc.layers.conv2d(
            self.res4d, kernel_size=1, num_outputs=256, scope='res4e_branch2a')
        self.res4e_branch2b = tc.layers.conv2d(
            self.res4e_branch2a, kernel_size=3, num_outputs=256, scope='res4e_branch2b')
        self.res4e_branch2c = tc.layers.conv2d(
            self.res4e_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4e_branch2c')
        self.res4e = tf.add(self.res4e_branch2c, self.res4d, name='res4e_add')
        self.res4e = tf.nn.relu(self.res4e, name='res4e')

        # Residual block 4f
        self.res4f_branch2a = tc.layers.conv2d(
            self.res4e, kernel_size=1, num_outputs=256, scope='res4f_branch2a')
        self.res4f_branch2b = tc.layers.conv2d(
            self.res4f_branch2a, kernel_size=3, num_outputs=256, scope='res4f_branch2b')
        self.res4f_branch2c = tc.layers.conv2d(
            self.res4f_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4f_branch2c')
        self.res4f = tf.add(self.res4f_branch2c, self.res4e, name='res4f_add')
        self.res4f = tf.nn.relu(self.res4f, name='res4f')



        # Residual block 5a
        self.res5a_branch2a_new = tc.layers.conv2d(
            self.res4f, kernel_size=1, num_outputs=512, scope='res5a_branch2a_new')
        self.res5a_branch2b_new = tc.layers.conv2d(
            self.res5a_branch2a_new, kernel_size=3, num_outputs=512, scope='res5a_branch2b_new')
        self.res5a_branch2c_new = tc.layers.conv2d(
            self.res5a_branch2b_new, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
        self.res5a_branch1_new = tc.layers.conv2d(
            self.res4f, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch1_new')
        self.res5a = tf.add(self.res5a_branch2c_new,
                            self.res5a_branch1_new, name='res5a_add')
        self.res5a = tf.nn.relu(self.res5a, name='res5a')
        self.res5a_heatmap1a = tf.layers.conv2d_transpose(
            self.res5a, kernel_size=4, filters=self.n_joints*2, activation=None, strides=2, padding='same', use_bias=False, name='res5a_heatmap1a')
        self.res5a_heatmap_bn = tc.layers.batch_norm(
            self.res5a_heatmap1a, scale=True, is_training=self.is_training, scope='res5a_heatmap_bn')
        self.heatmap_intermidiate2, self.z_heatmap_intermidiate2 = tf.split(tf.nn.relu(
            self.res5a_heatmap_bn, name='intermidiate2'), num_or_size_splits=2, axis=3)



        # Residual block 5b
        self.res5b_branch2a_new = tc.layers.conv2d(
            self.res5a, kernel_size=1, num_outputs=256, scope='res5b_branch2a_new')
        self.res5b_branch2b_new = tc.layers.conv2d(
            self.res5b_branch2a_new, kernel_size=3, num_outputs=128, scope='res5b_branch2b_new')
        self.res5b_branch2c_new = tc.layers.conv2d(
            self.res5b_branch2b_new, kernel_size=1, num_outputs=256, scope='res5b_branch2c_new')

        # Transpose Conv
        self.res5c_branch1a = tf.layers.conv2d_transpose(
            self.res5b_branch2c_new, kernel_size=4, filters=self.n_joints*3, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch1a')
        self.res5c_branch2a = tf.layers.conv2d_transpose(
            self.res5b_branch2c_new, kernel_size=4, filters=128, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch2a')
        self.bn5c_branch2a = tc.layers.batch_norm(
            self.res5c_branch2a, scale=True, is_training=self.is_training, scope='bn5c_branch2a')
        self.bn5c_branch2a = tf.nn.relu(self.bn5c_branch2a)

        self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z = tf.split(
            self.res5c_branch1a, num_or_size_splits=3, axis=3)
        self.res5c_branch1a_sqr = tf.multiply(
            self.res5c_branch1a, self.res5c_branch1a, name='res5c_branch1a_sqr')
        self.res5c_delta_x_sqr, self.res5c_delta_y_sqr, self.res5c_delta_z_sqr = tf.split(
            self.res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
        self.res5c_bone_length_sqr = tf.add(
            tf.add(self.res5c_delta_x_sqr, self.res5c_delta_y_sqr), self.res5c_delta_z_sqr)
        self.res5c_bone_length = tf.sqrt(self.res5c_bone_length_sqr)

        self.res5c_branch2a_feat = tf.concat([self.bn5c_branch2a, self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z, self.res5c_bone_length],
                                             axis=3, name='res5c_branch2a_feat')

        self.res5c_branch2b = tc.layers.conv2d(
            self.res5c_branch2a_feat, kernel_size=3, num_outputs=128, scope='res5c_branch2b',activation_fn=tf.nn.relu)
        self.res5c_branch2c = tf.layers.conv2d(
            self.res5c_branch2b, kernel_size=1, filters=self.n_joints*2, activation=None, use_bias=False, name='res5c_branch2c')
        
        self.heatmap, self.z_heatmap = tf.split(
            self.res5c_branch2c, num_or_size_splits=2, axis=3)

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.total_batch)

    def save(self, checkpoint_dir, step):
        model_name = "vnect.model"
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
        ds.loadH36M(64*50, mode='train', with_heatmap=True,
                    tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadH36M(109867, mode='valid', with_heatmap=False,
                        tApp=True, replace=False)
        Vnect = vnect()

    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(64*300, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadApe(64, mode='valid', tApp=True, replace=False)

        Vnect = vnect()

    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    # Vnect.train(ds, val_ds)
    train_total_batch = 109867//64
    Vnect.predict(val_ds, train_total_batch)
