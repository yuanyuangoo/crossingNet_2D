import cv2
import numpy as np
import os


class Image(object):
    def __init__(self, dataset, path):
        if dataset.upper() == 'SKATE':
            self.loadSKATE(path)
        elif dataset.upper() == 'H36M':
            self.loadH36M(path)
        self.data=[]
    '''
    loading module
    '''

    def loadSKATE(self, path):
        img = cv2.imread(path, 0)

        self.Data = np.asarray(img-127.5, np.float32)/127.5
        self.size2 = self.Data.shape
        return self.Data

    def loadH36M(self, path):
        if os.path.exists(path)==False:
            raise IOError('Can''t find the image {}'.format(path))

        img = cv2.imread(path, 0)

        self.Data = np.asarray(img-127.5, np.float32)/127.5
        self.size2 = self.Data.shape
        return self.Data
