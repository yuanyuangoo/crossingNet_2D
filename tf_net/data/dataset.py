from tqdm import tqdm
import pickle
import os
import numpy as np
import time
import sys
sys.path.append('./')
import util
from Image import Image
import globalConfig
from data.util import Frame
class Dataset(object):
    def __init__(self):
        self.dataset = globalConfig.dataset
        # print('initialized')

        if self.dataset == 'H36M':
            self.refPtIdx = [0]
            self.skel_num = 17
            self.centerPtIdx = 0
            self.with_pose = True
        elif self.dataset == 'SKATE':
            self.refPtIdx = [0]
            self.skel_num = 17
            self.centerPtIdx = 0
            self.with_pose = False
        elif self.dataset =='APE':
            self.skel_num=15
            self.centerPtIdx=2
            self.with_pose=True
            self.refPtIdx=[2]


        # self.h36m_base_path = globalConfig.h36m_base_path
        # self.h36m_frm_perfile = 10000  # the maximum number of frame to store in each file
        self.cache_base_path = globalConfig.cache_base_path

    def loadApe(self, Fsize, frmStartNum=0, mode='train', replace=False, tApp=False):
        from ape import APE
        '''
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''
        if not hasattr(self, 'frmList'):
            self.frmList = []
        if not tApp:
            self.frmList = []
        pickleCachePath = '{}ape_{}_{}.pkl'.format(
            self.cache_base_path, mode, Fsize)

        if os.path.isfile(pickleCachePath) and not replace:
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            # (self.frmList) += pickle.load(f)
            (self.frmList) += pickle.load(f)
            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            self.width=img.size2[0]
            self.height=img.size2[1]
            return self.frmList

        data=APE()
        imagespath, depthspath, skels,labels = data.readallData()
        if mode is "train":
            frmEndNum = len(depthspath)-1000
        else:
            frmStartNum=len(depthspath)-1000
            frmEndNum = len(depthspath)

        for frmIdx in tqdm(range(frmStartNum, int(frmEndNum/Fsize)*Fsize, int((frmEndNum-frmStartNum)/Fsize))):
            img = Image('APE', depthspath[frmIdx])
            img_RGB = Image('APE', imagespath[frmIdx], RGB=True)

            self.frmList.append(
                Frame(img, img_RGB, skels[frmIdx], labels[frmIdx], depthspath[frmIdx]))
        self.width=img.size2[0]
        self.height=img.size2[1]

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        pickle.dump((self.frmList), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('loaded with {} frames'.format(len(self.frmList)))

    def loadSkate(self, Fsize, frmStartNum=0, mode='train', replace=False, tApp=False):
        from skate import SKATE
        '''
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''
        if not hasattr(self, 'frmList'):
            self.frmList = []
        if not tApp:
            self.frmList = []
        pickleCachePath = '{}skate_{}_{}.pkl'.format(
            self.cache_base_path, mode, Fsize)

        if os.path.isfile(pickleCachePath) and not replace:
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            # (self.frmList) += pickle.load(f)
            (self.frmList) += pickle.load(f)
            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            return self.frmList

        data = SKATE(mode)
        frmEndNum =data.nSamples
        for frmIdx in tqdm(range(frmStartNum, int(frmEndNum/Fsize)*Fsize, int(frmEndNum/Fsize))):
            while True:
                [frmPath, label] = data.getImgName_onehotLabel(frmIdx)
                frmPath_rgb = data.getImgName_RGB(frmIdx)
                
                if mode=='train':
                    if os.path.exists(frmPath) and os.path.exists(frmPath_rgb):
                        break
                    else:
                        frmIdx = frmIdx+1
                else:
                    break
            
            # skel = np.asarray(data.getSkel(frmIdx))
            # if skel.shape == ():
            #     continue
            # skel.shape = (-1)

            img = Image('Skate', frmPath)
            img_RGB = Image('Skate', frmPath_rgb, RGB=True)

            self.frmList.append(
                Frame(img, img_RGB, frmPath, label=label))
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

    def loadH36M(self, Fsize, frmStartNum=0, mode='train', replace=False, tApp=False):
        from h36m import H36M
        '''
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''
        if not hasattr(self, 'frmList'):
            self.frmList = []
        if not tApp:
            self.frmList = []
        # fileIdx = int(frmStartNum / self.h36m_frm_perfile)
        pickleCachePath = '{}h36m_{}_{}.pkl'.format(
            self.cache_base_path, mode, Fsize)

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

        frmEndNum = data.nSamples

        for frmIdx in tqdm(range(frmStartNum, int(frmEndNum/Fsize)*Fsize, int(frmEndNum/Fsize))):
            while True:
                [frmPath, label] = data.getImgName_onehotLabel(frmIdx)
                frmPath_rgb = data.getImgName_RGB(frmIdx)
                if os.path.exists(frmPath) and os.path.exists(frmPath_rgb):
                    break
                else:
                    frmIdx = frmIdx+1
            
            skel = np.asarray(data.getSkel(frmIdx))
            if skel.shape == ():
                continue
            skel.shape = (-1)

            img = Image('H36M', frmPath)
            img_RGB = Image('H36M', frmPath_rgb, RGB=True)

            self.frmList.append(Frame(img, img_RGB, skel, label, frmPath))
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
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        # for i in range(0, 20000, 20000):
        ds.loadH36M(1024, mode='train', tApp=True, replace=True)

        val_ds = Dataset()
        # for i in range(0, 20000, 20000):
        val_ds.loadH36M(64, mode='valid', tApp=True, replace=True)

    elif globalConfig.dataset == 'Skate':
        val_ds = Dataset()
        val_ds.loadSkate(64, mode='valid', tApp=True, replace=True)

    elif globalConfig.dataset == 'APE':
        val_ds = Dataset()
        val_ds.loadApe(1024, mode='train', tApp=True, replace=False)
    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)
