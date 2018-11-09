import json
from sklearn.externals import joblib
import shutil
from numpy.random import RandomState
import time
import cv2
import sys
sys.path.append('./')
import globalConfig
sys.path.append('./data/')
from data.dataset import *
b1 = 0.5
noise_dim = 23
K = 1
class ImageGAN(object):
    def __init__(self, z_dim, batch_size=100, lr=0.0005, b1=0.5):
        self.z_dim = z_dim
        self.ratio = 1
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.z_std = 0.6  # used when z is normal distribution
        print ('ImageGAN is initialized with z_dim=%d' % self.z_dim)