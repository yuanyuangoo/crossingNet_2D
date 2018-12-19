import globalConfig
import ref
import pickle
import time
import os
from pprint import pprint
import random
# from util import Rnd, Flip, ShuffleLR
import cv2
import numpy as np
import sys
sys.path.append('../data/')


class SKATE:
    def __init__(self, split):
        self.root = '/media/hsh65/Portable/skate/'
        self.imgdir_origin = self.root+'input/'
        self.bacimgdir = self.root + 'binary/'
        self.bacimg_list = os.listdir(self.bacimgdir)
        self.nSamples = len(self.bacimg_list)

        if split == 'valid':
            self.imgdir_origin = self.root+'r/'
            self.bacimgdir = self.root + 'r/'
            self.bacimg_list = os.listdir(self.imgdir_origin)
            self.nSamples = len(self.bacimg_list)

    def getImgName_onehotLabel(self, idx):
        frmpath = self.bacimgdir+self.bacimg_list[idx]
        label = np.zeros((15))
        label[1] = 1
        return frmpath, label

    def getImgName_RGB(self, idx):
        frmpath = self.imgdir_origin+self.bacimg_list[idx]
        return frmpath
    


