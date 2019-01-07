from tqdm import tqdm
import pickle
import os
import numpy as np
import time
import sys
sys.path.append('./')
from Image import Image
import globalConfig
from data.util import *
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
            self.width=147
            self.height=110


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

            return self.frmList

        data=APE(mode)
        imagespath, depthspath, skels,labels = data.readallData()
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
    def loadH36M_expended(self, Fsize, frmStartNum=0, mode='train', replace=False, tApp=False):
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
        pickleCachePath_expanded = '{}h36m_{}_{}_{}.pkl'.format(
            self.cache_base_path, mode, "with_expanded", Fsize)

        if os.path.isfile(pickleCachePath_expanded) and not replace:
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath_expanded, 'rb')
            # (self.frmList) += pickle.load(f)
            (self.frmList) += pickle.load(f)
            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            return self.frmList


        pickleCachePath = '{}h36m_{}_{}.pkl'.format(
            self.cache_base_path, mode, Fsize)

        if os.path.isfile(pickleCachePath):
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            # (self.frmList) += pickle.load(f)
            (self.frmList) += pickle.load(f)
            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            # return self.frmList
        
        test_skel = np.load('samples_skel.out.npy')
        test_label = np.load('samples_label.out.npy')
        n_samples = len(self.frmList)
        img_path = '/media/hsh65/Portable/h36m/cache/model/pganR/H36M_dummy/params/samples_predicted/'

        for idx in tqdm(range(len(test_label))):
            label = test_label[idx]
            skel = test_skel[idx]
            frmPath_rgb = '{}/predict_{:04d}.png'.format(img_path, idx)

            img_RGB = Image('H36M', frmPath_rgb, RGB=True)
            img = []
            self.frmList.append(
                Frame(img, img_RGB, skel, label, frmPath_rgb))

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)

        f = open(pickleCachePath_expanded, 'wb')
        pickle.dump((self.frmList), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print("{}_writing completed".format(len(self.frmList)))

    def loadH36M_all(self, batch_idx, frmStartNum=0, mode='train', replace=False, tApp=False,with_background=False):
        if not hasattr(self, 'frmList'):
            self.frmList = []
        if not tApp:
            self.frmList = []
        with_back = ''
        if with_background:
            with_back = 'with_back'
        self.frmList = []
        from h36m import H36M
        data = H36M(mode)
        nSamples = data.nSamples
        nums_in_onebatch = (64*200)
        nbatch = nSamples//nums_in_onebatch
        print('Processing batch {} in {}'.format(batch_idx, nbatch))
        pickleCachePath = '{}h36m_{}_{}_{}_{}.pkl'.format(
            self.cache_base_path, mode, 'all', with_back, batch_idx)
        if os.path.isfile(pickleCachePath) and not replace:
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            # (self.frmList) += pickle.load(f)
            (self.frmList) += pickle.load(f)
            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            return self.frmList

        self.frmList = []
        frmStartNum = batch_idx*nums_in_onebatch
        frmEndNum = min((batch_idx+1)*nums_in_onebatch, nSamples)

        for frmIdx in tqdm(range(frmStartNum, frmEndNum)):
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
            img=[]
            if with_background:
                img = Image('H36M', frmPath)
            img_RGB = Image('H36M', frmPath_rgb, RGB=True)
            self.frmList.append(Frame(img, img_RGB, skel, label, frmPath))
        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        pickle.dump((self.frmList), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('loaded with {} frames'.format(len(self.frmList)))

    def loadH36M(self, Fsize, frmStartNum=0, mode='train', replace=False, tApp=False,with_background=False):
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

        with_back = ''
        if with_background:
            with_back = 'with_back'
        pickleCachePath = '{}h36m_{}_{}_{}.pkl'.format(
                self.cache_base_path, mode, Fsize,with_back)

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
            img=[]
            if with_background:
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
        # ds = Dataset()
        # ds.loadH36M_expended(64*50, mode='train',
        #             tApp=True, replace=True)

        val_ds = Dataset()
        val_ds.loadH36M(64, mode='valid',
                        tApp=True, replace=False, with_background=False)
        
    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(64*300, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadApe(64, mode='valid', tApp=True, replace=False)

    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)
