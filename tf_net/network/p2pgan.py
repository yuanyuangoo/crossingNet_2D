import tensorflow as tf
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data.util import *
from data.dataset import *
from data.layers import *
import globalConfig
EPS = 1e-12
CROP_SIZE = 128
class P2PGAN(object):
    def __init__(self, mode='train', output_dir='sample', checkpoint_dir="./checkpoint",
                 epoch=200, aspect_ratio=1, batch_size=64, ngf=64, ndf=64, scale_size=286,
                 flip=True, lr=0.0002, beta1=0.5, l1_weight=100, gan_weight=1):
        a = 1


if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        for i in range(0, 20000, 20000):
            ds.loadH36M(i, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        for i in range(0, 20000, 20000):
            val_ds.loadH36M(i, mode='valid', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)

    gan = P2PGAN()
    gan.train(ds, val_ds)

