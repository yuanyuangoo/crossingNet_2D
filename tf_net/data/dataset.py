import time
import sys
sys.path.append('./')
from data.util import Frame
import globalConfig
import h5py
import numpy as np
from numpy.matlib import repmat
from numpy.linalg import svd, det
from Image import Image
import os
import pickle
from tqdm import tqdm
import util
from h36m import H36M
import sys
sys.path.append('./')


class Dataset(object):
    def __init__(self):
        self.dataset = globalConfig.dataset
        # print('initialized')

        if self.dataset == 'H36M':
            self.refPtIdx = [0]
            self.skel_num = 17
            self.centerPtIdx = 0
            self.with_pose = True

        self.h36m_base_path = globalConfig.h36m_base_path
        self.h36m_frm_perfile = 10000  # the maximum number of frame to store in each file
        self.cache_base_path = globalConfig.cache_base_path

    def loadH36M(self, frmStartNum, mode='train', replace=False, tApp=False):
        '''
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''
        if not hasattr(self, 'frmList'):
            self.frmList = []
        if not tApp:
            self.frmList = []
        fileIdx = int(frmStartNum / self.h36m_frm_perfile)
        pickleCachePath = '{}h36m_{}_{}.pkl'.format(self.cache_base_path,
                                                     mode, fileIdx)

        if os.path.isfile(pickleCachePath) and not replace:
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            # (self.frmList) += pickle.load(f)
            (self.frmList) += pickle.load(f)
            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            return self.frmList

        data = H36M(mode)
        if frmStartNum >= data.nSamples:
            raise ValueError(
                'invalid start frame, shoud be lower than {}'.format(data.nSamples))

        frmStartNum = fileIdx*self.h36m_frm_perfile
        frmEndNum = min(frmStartNum+self.h36m_frm_perfile, data.nSamples)

        print('frmStartNum={}, frmEndNum={}, fileIdx={}'.format(frmStartNum,
                                                                frmEndNum,
                                                                fileIdx))

        for frmIdx in tqdm(range(frmStartNum, frmEndNum)):
            [frmPath, label] = data.getImgName_onehotLabel(frmIdx)

            skel = np.asarray(data.getSkel(frmIdx))
            if skel.shape == ():
                continue
            skel.shape = (-1)

            img = Image('H36M', frmPath)
            self.frmList.append(Frame(img, skel, label))
            # self.frmList[-1].saveOnlyForTrain()

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        pickle.dump((self.frmList), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('loaded with {} frames'.format(len(self.frmList)))

    '''
    interface to neural network, used for training
    '''


if __name__ == '__main__':
    dataset = globalConfig.dataset

    if dataset == 'H36M':
        ds = Dataset()
        for i in range(0, 150000, 10000):
            print(i)
            ds.loadH36M(i, tApp=True, replace=False, mode='valid')
        for i in range(0, 3000000, 10000):
            print(i)
            ds.loadH36M(i, tApp=True, replace=False, mode='Train')