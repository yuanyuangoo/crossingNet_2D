'''
import the sequence and pre-process it
'''
import sys

sys.path.append('./')
import h5py
import globalConfig

import progressbar as pb
import numpy as np
from numpy.matlib import repmat
from numpy.linalg import svd, det
from Image import Image
from util import Frame, Camera
from geometry import Quaternion, Matrix33
import os
import pickle
import time

# for pickle module
import sys
import util


class Dataset(object):
    '''
    base type to load dataset and as interface to the training
    '''

    def __init__(self):
        self.dataset = globalConfig.dataset
        print('initialized')

        if self.dataset == 'H36M':
            self.refPtIdx = [1]
            self.skel_num = 17
            self.centerPtIdx = 1

        self.cache_base_path = globalConfig.cache_base_path
        self.msra_base_path = globalConfig.msra_base_path
        self.msra_pose_list = '1  2  3  4  5  6  7  8  9  I  IP  L  MP  RP  T  TIP  Y'.split()
        self.nyu_base_path = globalConfig.nyu_base_path
        self.nyu_frm_perfile = 20000  # the maximum number of frame to store in each file
        self.icvl_base_path = globalConfig.icvl_base_path
        self.h36m_base_path = globalConfig.h36m_base_path
        self.h36m_frm_perfile = 200  # the maximum number of frame to store in each file

    def loadH36M(self, frmStartNum, mode='train', replace=False, tApp=False):
        '''
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''

        NamesPath = '/'.join([self.h36m_base_path, 'annot/train_images.txt'])

        if mode == 'train':

            annotPath='/'.join([self.h36m_base_path,'annot/train.mat'])
            # NamesPath = '/'.join([self.h36m_base_path,
            #                       'annot/train_images.txt'])
            # Hdf5Path = '/'.join([self.h36m_base_path, 'annot/train.h5'])

        elif mode == 'test':
            annotPath='/'.join([self.h36m_base_path,'annot/valid.mat'])

            # NamesPath = '/'.join([self.h36m_base_path,
            #                       'annot/test_images.txt'])
            # Hdf5Path = '/'.join([self.h36m_base_path, 'annot/test.h5'])

        # with open(NamesPath, 'r') as f:
        #     Names = f.readlines()
        # print(len(Names))
        # determine the start and end frame
        from scipy.io import loadmat
        annot = loadmat(annotPath)['annot']
        imgname = annot['imgname'][0][0]

        if frmStartNum >= len(imgname):
            raise ValueError(
                'invalid start frame, shoud be lower than {}'.format(len(imgname)))
        fileIdx = int(frmStartNum / self.h36m_frm_perfile)
        frmStartNum = fileIdx*self.h36m_frm_perfile
        frmEndNum = min(frmStartNum+self.h36m_frm_perfile, len(imgname))

        print('frmStartNum={}, frmEndNum={}, fileIdx={}'.format(frmStartNum,
                                                                frmEndNum,
                                                                fileIdx))
        if not hasattr(self, 'frmList'):
            self.frmList = []
        if not tApp:
            self.frmList = []

        pickleCachePath = '{}/h36m_{}_{}.pkl'.format(self.cache_base_path,
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

        # f = h5py.File(Hdf5Path, 'r')
        # a_group_key = list(f.keys())[0]
        # skels = list(f[a_group_key])

        pbar = pb.ProgressBar(maxval=frmEndNum-frmStartNum,
                              widgets=['Loading H36M | ', pb.Percentage(), pb.Bar()])
        pbar.start()
        pbIdx = 0

        skels = annot['S_glob'][0][0]
        P2ds = annot['part'][0][0]

        cam=annot['cam'][0][0]
        for frmIdx in range(frmStartNum, frmEndNum):
            frmPath = os.path.join(
                self.h36m_base_path+'/images', imgname[frmIdx][0][0][:]+'.jpg')
            skel = np.asarray(skels[frmIdx])

            p2d = np.asarray(P2ds[frmIdx])
            # skel.shape = (-1)
            Rotation=cam[frmIdx][0]['R'][0][0]
            img = Image('H36M', frmPath)
            self.frmList.append(Frame(img, skel, Rotation, p2d))
            self.frmList[-1].saveOnlyForTrain()
            pbar.update(pbIdx)
            pbIdx += 1
        pbar.finish()

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        pickle.dump((self.frmList), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('loaded with {} frames'.format(len(self.frmList)))

    def normTranslation(self, origin_pt_idx=None):
        if origin_pt_idx is None:
            origin_pt_idx = self.centerPtIdx
        if not hasattr(self, 'frmList') or self.frmList == None:
            return

        if not hasattr(self.frmList[0], 'norm_skel'):
            return

        vec_dim = 3*self.skel_num
        for frm in self.frmList:
            frm.origin =\
                frm.norm_skel[origin_pt_idx*3:origin_pt_idx*3+3].copy()
            origin = repmat(frm.origin, 1, self.skel_num)
            origin.shape = (vec_dim,)
            frm.norm_skel -= origin

    def normRotation(self, tmpSkel=None, refPtIdx=None):
        '''tmpSkel: normalize every palm pose to the tmpSkel pose (template skeleton)
           refPtIdx: indexes of joints on the palm
        '''
        if tmpSkel is None:
            tmpSkel = self.frmList[0].norm_skel

        if refPtIdx is None:
            refPtIdx = self.refPtIdx
        refIdx = []
        for idx in refPtIdx:
            refIdx += [idx*3, idx*3+1, idx*3+2]

        keep_list = set(range(3*self.skel_num)).\
            difference(set(refIdx+range(self.centerPtIdx, self.centerPtIdx+3)))
        keep_list = list(keep_list)

        temp = tmpSkel[refIdx].copy()
        temp.shape = (-1, 3)

        for frm in self.frmList:
           model = frm.norm_skel[refIdx]
           model.shape = (-1, 3)

           R = np.zeros((3, 3), np.float32)
           for vt, vm in zip(temp, model):
               R = R + np.dot(vm.reshape(3, 1), vt.reshape(1, 3))

           U, s, V = svd(R, full_matrices=True)
           R = np.dot(V.transpose(), U.transpose())
           frm.quad = Quaternion(R)
           frm.norm_skel.shape = (-1, 3)
           frm.norm_skel = np.dot(R, frm.norm_skel.transpose())
           frm.norm_skel = frm.norm_skel.flatten('F')
           # frm.norm_skel = frm.norm_skel[keep_list]

    def skelNum(self):
        if self.frmList is None or len(self.frmList) == 0:
            raise ValueError('frameList is empty')
        return len(self.frmList[0].norm_skel)

    def frmToNp(self):
        '''
        prepare the training samples for training the neural network
        normalize the input and output data
        '''
        if self.frmList is None or len(self.frmList) == 0:
            raise ValueError('frameList is empty')

        frmNum = len(self.frmList)
        jntNum = len(self.frmList[0].norm_skel)

        # normalized
        self.x_norm = np.zeros(
            (frmNum, 1, Image.size2[1], Image.size2[0]), np.float32)
        self.y_norm = np.zeros((frmNum, jntNum), np.float32)
        self.pose_orig = np.zeros((frmNum, 3), np.float32)
        self.pose_trans = np.zeros((frmNum, 3, 3), np.float32)
        self.pose_com = np.zeros((frmNum, 3), np.float32)

        for i, frm in enumerate(self.frmList):
            if np.any(np.isnan(frm.norm_dm)):
                self.x_norm[i] = self.x_norm[max(i-1, 0)]
                self.y_norm[i] = self.y_norm[max(i-1, 0)]
                self.pose_orig[i] = self.pose_orig[max(i-1, 0)]
                self.pose_trans[i] = self.pose_trans[max(i-1, 0)]
                self.pose_com[i] = self.pose_com[max(i-1, 0)]
                continue
            if np.any(np.isnan(frm.norm_skel)):
                self.x_norm[i] = self.x_norm[max(i-1, 0)]
                self.y_norm[i] = self.y_norm[max(i-1, 0)]
                self.pose_orig[i] = self.pose_orig[max(i-1, 0)]
                self.pose_trans[i] = self.pose_trans[max(i-1, 0)]
                self.pose_com[i] = self.pose_com[max(i-1, 0)]
                continue

            # reverse the background color to background
            frm.norm_dm[frm.norm_dm == frm.norm_dm.min()] = 0.5
            frm.norm_dm[frm.norm_dm != 0.5] -= frm.norm_dm.min()+0.5
            frm.norm_dm *= np.float32(2)

            self.x_norm[i, 0] = frm.norm_dm
            self.y_norm[i] = frm.norm_skel
            if hasattr(frm, 'origin'):
                self.pose_orig[i] = frm.origin
            self.pose_trans[i] = frm.trans
            self.pose_com[i] = frm.com3D

        print('x_norm range: {} to {}'.format(self.x_norm.min(),
                                              self.x_norm.max()))
        return self.x_norm, self.y_norm


if __name__ == '__main__':
    dataset = globalConfig.dataset
    
    if dataset == 'H36M':
        ds = Dataset()
        for i in range(0, 20000, 20000):
            ds.loadH36M(i, tApp=True)
