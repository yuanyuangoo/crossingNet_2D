import scipy.optimize
from forwardRender import ForwardRender
import cv2
import time
import shutil
import os
import json
import numpy.linalg
from numpy.random import RandomState
import tensorflow as tf
from collections import namedtuple
import sys
sys.path.append('./')
import globalConfig
from data.dataset import *


class GanRender(ForwardRender):
    DisErr = namedtuple('disErr', ['gan', 'est', 'metric'])
    GenErr = namedtuple('genErr', ['gan', 'recons', 'metric'])
    golden_max = 1.0

    def __init__(self, x_dim, rndGanInput=False, metricCombi=False):
        super(GanRender, self).__init__(x_dim)
        self.rndGanInput = rndGanInput
        self.metricCombi = metricCombi

    def train(self, nepoch=None, train_dataset=None, valid_dataset=None, desc='dummy'):
        cache_dir = os.path.join(
            globalConfig.model_dir, 'gan_render/%s_%s' % (globalConfig.dataset, desc))
        model_dir = os.path.join(cache_dir, 'pretrained_model')
        cache_dir = os.path.join(cache_dir, desc)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        log_path = os.path.join(cache_dir, 'log.txt')
        flog = open(log_path, 'w')
        flog.close()

        img_dir = os.path.join(cache_dir, 'img')
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)
        param_dir = os.path.join(cache_dir, 'params')
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

        self.pose_vae.load(globalConfig.vae_pretrain_path)
        self.image_gan.load(globalConfig.gan_pretrain_path)

        train_size = len(train_dataset.frmList)
        self.x_hat = tf.placeholder(
            tf.float32, shape=[None, self.dim_x], name='input_pose')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for frm in valid_dataset.frmList:
            test_data.append(frm.skel)
            test_labels.append(frm.label)

        for frm in train_dataset.frmList:
            train_data.append(frm.skel)
            train_labels.append(frm.label)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        train_data = train_data/max(-1*train_data.min(), train_data.max())
        test_data = test_data/max(-1*test_data.min(), test_data.max())
        VALIDATION_SIZE = 5000  # Size of the validation set.

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, :]
        validation_labels = train_labels[:VALIDATION_SIZE, :]

        train_data = train_data[VALIDATION_SIZE:, :]
        train_labels = train_labels[VALIDATION_SIZE:, :]
        train_total_data = np.concatenate(
            (train_data, train_labels), axis=1)
        NUM_LABELS = 15
        train_data_ = train_total_data[:, :-NUM_LABELS]
        train_size = train_total_data.shape[0]
        valid_data = np.concatenate(
            (validation_data, validation_labels), axis=1)
        test_data = np.concatenate(
            (test_data, test_labels), axis=1)

        print('[ganRender] begin training loop')
        seed = 42
        np_rng = RandomState(seed)
        train_size = train_total_data.shape[0]
        n_samples = train_size
        total_batch = int(n_samples / self.batch_size)
        with tf.Session() as sess:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            for epoch in tqdm(range(nepoch)):
                gen_errs, dis_errs = np.zeros((3,)), np.zeros((3,))
                gen_err, dis_err, nupdates = 0, 0, 0
                for i in range(total_batch):
                    offset = (i * self.batch_size) % (n_samples)
                    batch_xs_input = train_data_[
                        offset:(offset + self.batch_size), :]
                    batch_xs_target = batch_xs_input
                    vae_noises = np_rng.normal(0, 0.05,
                                               (self.batch_size, self.pose_z_dim)
                                               ).astype(np.float32)
                    gan_noises = GanRender.rndCvxCombination(np_rng,
                                                             src_num=self.batch_size,
                                                             tar_num=self.batch_size,
                                                             sel_num=5)
                    if epoch < 21:
                        real_render_var = sess.run(
                            (self.train_op, self.loss),
                            feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 0.9})

                        fake_render_var = sess.run(
                            (self.train_op, self.loss),
                            feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 0.9})

                        recons_loss = (real_render_var - fake_render_var)**2
                        recons_loss = tf.clip_by_value(
                            recons_loss, 0, self.golden_max)
                        gen_err = tf.reduce_mean(recons_loss)

                        gen_errs += np.array([0., 1., 0.])*gen_err

                        est_err, metr_err = sess.run(
                            (self.train_op, self.loss),
                            feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                        dis_errs += np.array([0., 1., 0.])*est_err +\
                            np.array([0., 0., 1.])*metr_err
                        nupdates += 1
                        continue
                    if self.rndGanInput:
                        dis_errs +=\
                            sess.run(
                                (self.train_op, self.loss),
                                feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                        gen_errs += \
                            sess.run(
                                (self.train_op, self.loss),
                                feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    else:
                        dis_errs += \
                            sess.run(
                                (self.train_op, self.loss),
                                feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                        gen_errs += \
                            sess.run(
                                (self.train_op, self.loss),
                                feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    nupdates += 1

                    dis_errs /= nupdates
                    gen_errs /= nupdates
                    dis_errs = self.DisErr(*tuple(dis_errs))
                    gen_errs = self.GenErr(*tuple(gen_errs))
                    print('disErr: {}'.format(dis_errs))
                    print('genErr: {}'.format(gen_errs))
                    flog = open(log_path, 'a')
                    flog.write('epoch {}s\n'.format(epoch))
                    flog.write(json.dumps((dis_errs, gen_errs))+'\n')
                    flog.close()
                    if epoch % 10 == 0 and valid_dataset is not None:
                        idx = 0
                for i in range(total_batch):
                    offset = (i * self.batch_size) % (n_samples)
                    batch_xs_input = train_data_[
                        offset:(offset + self.batch_size), :]
                    batch_xs_target = batch_xs_input
                    noise = np.zeros((1, self.pose_z_dim), np.float32)
                    reco_image = sess.run(
                        (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    reco_pose = sess.run(
                        (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    pose = self.resumePose(reco_pose[0], train_data[0]
                                           )

                    fake_img = self.visPair(reco_image[0],
                                            pose,
                                            50.0)

                    pose = self.resumePose(train_data[0],
                                           train_data[0])
                    real_img = self.visPair(train_data[0],
                                            pose,
                                            50.0)
                    est_z = sess.run(
                        (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    est_z.shape = (23,)
                    est_z, est_orig = est_z[:20], est_z[20:]
                    est_z.shape = (1, 20)
                    est_orig.shape = (1, 3)
                    est_pose = sess.run(
                        (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    est_image = sess.run(
                        (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_xs_input, self.x: batch_xs_target, self.keep_prob: 1})
                    pose = self.resumePose(est_pose[0],
                                           est_orig[0])
                    est_img = self.visPair(est_image[0],
                                           pose,
                                           50.0)
                    com_img = self.visPair(train_data[0],
                                           pose)
                    recons_img = np.hstack(
                        (real_img, fake_img, est_img, com_img))
                    cv2.imwrite(os.path.join(img_dir, '%d_%d.jpg' % (epoch, idx)),
                                recons_img.astype('uint8'))
                    idx += 1

            if epoch % 10 == 0:
                self.save(os.path.join(param_dir, '-1'), epoch)

            if epoch % 100 == 0:
                self.save(os.path.join(param_dir, '%d' % epoch), epoch)

                
    @classmethod
    def rndCvxCombination(cls, rng, src_num, tar_num, sel_num):
        # generate tar_num random convex combinations from src_num point, every
        # time only sel_num from the src_num are used
        if sel_num > src_num:
            raise ValueError('sel_num %d should less then src_num %d' % (sel_num,
                                                                         src_num))
        m = np.zeros((tar_num, src_num))
        for s in m:
            sel = rng.choice(src_num, sel_num)
            s[sel] = rng.uniform(0, 1, (sel_num,))
            s /= s.sum()
        return m.astype(np.float32)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            globalConfig.dataset, self.batch_size,
        )

    def save(self, checkpoint_dir, step):
        model_name = "GanRender.model"
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
        for i in range(0, 20000, 20000):
            ds.loadH36M(i, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadH36M(i, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    print('validation length = %d' % len(val_ds.frmList))
    skel_num = len(val_ds.frmList[0].skel)
    print('skel_num=%d' % skel_num)
    render = GanRender(x_dim=skel_num, rndGanInput=True, metricCombi=False)

    desc = 'pretrained'
    render.train(101, ds, val_ds,
                 desc=desc)
    # render.test(val_ds, desc=desc, modelIdx='-1')
