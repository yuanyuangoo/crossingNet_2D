import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import ConcatLayer, Conv2d, InputLayer, MaxPool2d
import os
import time
from six.moves import xrange
import sys
sys.path.append('./')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import globalConfig
from data.dataset import *
from data.util import *
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

# tf.truncated_normal_initializer(stddev=0.01)
W_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(value=0.0)
weight_decay_factor = 5e-4

class itp(object):
    def __init__(self, batch_size=64, input_height=128, input_width=128, dim_z=17*3, dataset_name='H36M', checkpoint_dir="./checkpoint", sample_dir="samples",
                 learning_rate=0.0002, beta1=0.5, epoch=300, reuse=False):
        self.sample_dir = os.path.join(
            globalConfig.p2i_pretrain_path, sample_dir)

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta1 = beta1
        self.input_height = input_height
        self.input_width = input_width
        self.dim_z = dim_z
        self.net, self.total_loss, self.log_tensors = self.make_model(
            *one_element, is_train=True, reuse=False)



    def model(self, x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None, data_format='channels_last'):
        """Defines the entire pose estimation model."""

        def _conv2d(x, c, filter_size, strides, act, padding, name):
            return Conv2d(
                x, c, filter_size, strides, act, padding, W_init=W_init, b_init=b_init, name=name, data_format=data_format)

        def _maxpool2d(x, name):
            return MaxPool2d(x, (2, 2), (2, 2), padding='SAME', name=name, data_format=data_format)

        def concat(inputs, name):
            if data_format == 'channels_last':
                concat_dim = -1
            elif data_format == 'channels_first':
                concat_dim = 1
            else:
                raise ValueError('invalid data_format: %s' % data_format)
            return ConcatLayer(inputs, concat_dim, name=name)

        def state1(cnn, n_pos, mask_miss1, mask_miss2, is_train):
            """Define the first stage of openpose."""
            with tf.variable_scope("stage1/branch1"):
                b1 = _conv2d(cnn, 128, (3, 3), (1, 1),
                             tf.nn.relu, 'SAME', 'c1')
                b1 = _conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c2')
                b1 = _conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c3')
                b1 = _conv2d(b1, 128, (1, 1), (1, 1),
                             tf.nn.relu, 'VALID', 'c4')
                b1 = _conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', 'confs')
                if is_train:
                    b1.outputs = b1.outputs * mask_miss1
            with tf.variable_scope("stage1/branch2"):
                b2 = _conv2d(cnn, 128, (3, 3), (1, 1),
                             tf.nn.relu, 'SAME', 'c1')
                b2 = _conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c2')
                b2 = _conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c3')
                b2 = _conv2d(b2, 128, (1, 1), (1, 1),
                             tf.nn.relu, 'VALID', 'c4')
                b2 = _conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', 'pafs')
                if is_train:
                    b2.outputs = b2.outputs * mask_miss2
            return b1, b2

        def stage2(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, scope_name):
            """Define the archuecture of stage 2 and so on."""
            with tf.variable_scope(scope_name):
                net = concat([cnn, b1, b2], 'concat')
                with tf.variable_scope("branch1"):
                    b1 = _conv2d(net, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c1')
                    b1 = _conv2d(b1, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c2')
                    b1 = _conv2d(b1, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c3')
                    b1 = _conv2d(b1, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c4')
                    b1 = _conv2d(b1, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c5')
                    b1 = _conv2d(b1, 128, (1, 1), (1, 1),
                                 tf.nn.relu, 'VALID', 'c6')
                    b1 = _conv2d(b1, n_pos, (1, 1), (1, 1),
                                 None, 'VALID', 'conf')
                    if is_train:
                        b1.outputs = b1.outputs * maskInput1
                with tf.variable_scope("branch2"):
                    b2 = _conv2d(net, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c1')
                    b2 = _conv2d(b2, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c2')
                    b2 = _conv2d(b2, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c3')
                    b2 = _conv2d(b2, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c4')
                    b2 = _conv2d(b2, 128, (3, 3), (1, 1),
                                 tf.nn.relu, 'SAME', 'c5')
                    b2 = _conv2d(b2, 128, (1, 1), (1, 1),
                                 tf.nn.relu, 'VALID', 'c6')
                    b2 = _conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', 'pafs')
                    if is_train:
                        b2.outputs = b2.outputs * maskInput2
            return b1, b2

        def vgg_network(x):
            x = x - 0.5
            # input layer
            net_in = InputLayer(x, name='input')
            # conv1
            net = _conv2d(net_in, 64, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv1_1')
            net = _conv2d(net, 64, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv1_2')
            net = _maxpool2d(net, 'pool1')
            # conv2
            net = _conv2d(net, 128, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv2_1')
            net = _conv2d(net, 128, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv2_2')
            net = _maxpool2d(net, 'pool2')
            # conv3
            net = _conv2d(net, 256, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv3_1')
            net = _conv2d(net, 256, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv3_2')
            net = _conv2d(net, 256, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv3_3')
            net = _maxpool2d(net, 'pool3')
            # conv4
            net = _conv2d(net, 512, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv4_1')
            net = _conv2d(net, 512, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv4_2')
            net = _conv2d(net, 256, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv4_3')
            net = _conv2d(net, 128, (3, 3), (1, 1),
                          tf.nn.relu, 'SAME', 'conv4_4')

            return net

        with tf.variable_scope('model', reuse):
            ## Feature extraction part
            cnn = vgg_network(x)
            b1_list = []
            b2_list = []
            ## stage 1
            b1, b2 = state1(cnn, n_pos, mask_miss1, mask_miss2, is_train)
            b1_list.append(b1)
            b2_list.append(b2)

            ## stage 2 ~ 6
            # for i in range(2, 7):
            # TODO: fix indent here and the names in npz
            with tf.variable_scope("stage1/branch2"):
                for i in [5, 6]:  # only 3 stage in total
                    b1, b2 = stage2(cnn, b1, b2, n_pos, mask_miss1,
                                    mask_miss2, is_train, scope_name='stage%d' % i)
                    b1_list.append(b1)
                    b2_list.append(b2)

            net = tl.layers.merge_networks([b1, b2])
            return cnn, b1_list, b2_list, net

    def make_model(self, img, results, mask, is_train=True, reuse=False):
        n_pos = self.dim_z/3
        confs = results[:, :, :, :n_pos]
        pafs = results[:, :, :, n_pos:]
        m1 = tf_repeat(mask, [1, 1, 1, n_pos])
        m2 = tf_repeat(mask, [1, 1, 1, n_pos * 2])

        cnn, b1_list, b2_list, net = self.model(img, n_pos, m1, m2, is_train, reuse)

        # define loss
        losses = []
        last_losses_l1 = []
        last_losses_l2 = []
        stage_losses = []

        for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
            loss_l1 = tf.nn.l2_loss((l1.outputs - confs) * m1)
            loss_l2 = tf.nn.l2_loss((l2.outputs - pafs) * m2)

            losses.append(tf.reduce_mean([loss_l1, loss_l2]))
            stage_losses.append(loss_l1 / self.batch_size)
            stage_losses.append(loss_l2 / self.batch_size)

        last_conf = b1_list[-1].outputs
        last_paf = b2_list[-1].outputs
        last_losses_l1.append(loss_l1)
        last_losses_l2.append(loss_l2)
        l2_loss = 0.0

        for p in tl.layers.get_variables_with_name('kernel', True, True):
            l2_loss += tf.contrib.layers.l2_regularizer(weight_decay_factor)(p)
        total_loss = tf.reduce_sum(losses) / self.batch_size + l2_loss

        log_tensors = {'total_loss': total_loss,
                       'stage_losses': stage_losses, 'l2_loss': l2_loss}
        net.cnn = cnn
        net.img = img  # net input
        net.last_conf = last_conf  # net output
        net.last_paf = last_paf  # net output
        net.confs = confs  # GT
        net.pafs = pafs  # GT
        net.m1 = m1  # mask1, GT
        net.m2 = m2  # mask2, GT
        net.stage_losses = stage_losses
        net.l2_loss = l2_loss
        return net, total_loss, log_tensors
if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        ds.loadH36M(1024, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=False)
        openpose = itp(dim_z=17*3)

    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(64*300, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadApe(64, mode='valid', tApp=True, replace=False)

        openpose = itp(dim_z=15*3)

    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    openpose.train(ds, val_ds)
