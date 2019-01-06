import cv2
import numpy as np
import os


class Image(object):
    def __init__(self, dataset, path, RGB=False):
        if dataset.upper() == 'SKATE':
            self.loadSKATE(path,RGB=RGB)
        elif dataset.upper() == 'H36M':
            self.loadH36M(path, RGB=RGB)
        elif dataset.upper() == 'APE':
            self.loadAPE(path, RGB=RGB)
        # self.data = []
    '''
    loading module
    '''

    def loadSKATE(self, path, RGB=False):
        if os.path.exists(path) == False:
            print(path)
            raise IOError('Can''t find the image {}'.format(path))

        if RGB:
            img = cv2.imread(path)
        else:
            img = cv2.imread(path, 0)
        w = img.shape[0]
        h = img.shape[1]
        if RGB:
            out = 255*np.ones((max(w, h), max(w, h), 3))
            out[0:w, 0:h, :] = img
        else:
            out = 255*np.ones((max(w, h), max(w, h)))
            out[0:w, 0:h] = img

        out = cv2.resize(out, (128, 128))
        out = out.reshape(128, 128, -1)
        
        self.Data = np.asarray(out-127.5, np.float32)/127.5
        self.size2 = self.Data.shape
        return self.Data

    def loadH36M(self, path, RGB=False):
        if os.path.exists(path) == False:
            raise IOError('Can''t find the image {}'.format(path))

        if RGB:
            img = cv2.imread(path)
        else:
            img = cv2.imread(path, 0)
            
        img = img.reshape(128, 128, -1)
        
        self.Data = np.asarray(img-127.5, np.float32)/127.5
        self.size2 = self.Data.shape
        return self.Data
        

    def loadAPE(self, path, RGB=False):
        if os.path.exists(path) == False:
            raise IOError('Can''t find the image {}'.format(path))

        if RGB:
            img = cv2.imread(path)
        else:
            img = cv2.imread(path, 0)

        w = img.shape[0]
        h = img.shape[1]
        # ratio = np.sqrt((w*h)/(128*128))
        # w=int(w/ratio)
        # h=int(h/ratio)
        if RGB:
            out = 255*np.ones((max(w, h), max(w, h), 3))
            out[0:w, 0:h, :] = img
        else:
            out = 255*np.ones((max(w, h), max(w, h)))
            out[0:w, 0:h] = img

        out = cv2.resize(img, (128, 128))
        out = out.reshape(128, 128, -1)
        
        self.Data = np.asarray(out-127.5, np.float32)/127.5
        self.size = self.Data.shape
        return self.Data
