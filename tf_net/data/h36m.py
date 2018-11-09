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
import ref
print(os.getcwd())
import globalConfig


class H36M:
    def __init__(self, split):
        # print('==>initializing 3D data.')
        annot = {}

        #tags=['S','center','index','normalize','part',
        #'person','scale','torsoangle','visible','zind']
        #f=h5py.File('/media/a/D/datasets/h36m/annot/'+split+'.h5','r')
        #fortagintags:
        #annot[tag]=np.asarray(f[tag]).copy()
        #f.close()
        #annot['imgname']={}
        #self.nSamples=len(annot['scale'])

        #withopen('/media/a/D/datasets/h36m/annot/'+split+'_images.txt','r')asf:
        #foriinrange(self.nSamples):
        #annot['imgname'][i]=f.readline()[:-1]

        annotPath = os.path.join(
            globalConfig.h36m_base_path, 'annot', split+'.mat')
        from scipy.io import loadmat
        annot = loadmat(annotPath)['annot']
        self.nSamples = len(annot['imgname'][0][0])
        self.annot = annot

        self.root = 0
        self.split = split

        # read Bounding Box
        bbpath = os.path.join(globalConfig.h36m_base_path, 'annot', 'bbox.txt')
        bbfile = open(bbpath, 'r')
        bbox = dict()
        for line in bbfile:
            filename = line[line.find('S'):line.find(',')]
            lx = int(line[line.find('lx')+3:line.find(',', line.find('lx'))])
            ly = int(line[line.find('ly')+3:line.find(',', line.find('ly'))])
            rx = int(line[line.find('rx')+3:line.find(',', line.find('rx'))])
            ry = int(line[line.find('ry')+3:-1])
            bbox[filename] = [lx, ly, rx, ry]

        self.bbox = bbox

        print('Loaded 3D {} samples'.format(self.nSamples))

    def getPart3D(self, index):
        pts_3d = self.annot['S_glob'][0][0][index].copy()
        return pts_3d

    def getPart2D(self, index):
        pts_2d = self.annot['part'][0][0][index].copy()
        return pts_2d

    def getImgName_onehotLabel(self, index):
        imgname = ref.h36mImgDir + \
            self.annot['imgname'][0][0][index][0][0][:]+'.jpg'
        label = self.getOneHotedLabel(imgname)
        return imgname, label

    def getImgName_Label(self, index):
        imgname = ref.h36mImgDir + \
            self.annot['imgname'][0][0][index][0][0][:]+'.jpg'
        label = self.getLabel(imgname)
        return imgname, label

    def Crop(self, p2d, bbox, res=128.0):
        ratio = max(abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1]))/res
        p2d_croped = (p2d-np.array([bbox[1], bbox[0]]).reshape(2, 1))/ratio
        return p2d_croped

    def getLabel(self, imgname):
        index = None
        for key, tag in ref.tags.items():
            if tag in imgname:
                index = ref.actions.index(key)
        if index == None:
            index = len(ref.actions)-1
        return index

    def getOneHotedLabel(self, imgname):
        index = None
        for key, tag in ref.tags.items():
            if tag in imgname:
                index = ref.actions.index(key)
        if index == None:
            index = len(ref.actions)-1
        return ref.oneHoted[index, :]

    def getAction(self, oneHot):
        return ref.actions[np.argmax(oneHot)]

    def getSkel(self, index):
        R = self.getRotation(index)
        P2d = self.getPart2D(index)
        imgname = self.getImgName_Label(index)[0]
        imgname = imgname[len(ref.h36mImgDir):]
        P2d_croped = self.Crop(P2d, self.bbox[imgname])
        P2d_centered = (P2d_croped.T-P2d_croped.T[self.root]).T
        P3d = self.getPart3D(index)

        P3d_roted = np.dot(R, P3d)
        P3d_roted_centered = (P3d_roted.T-(P3d_roted[:, self.root])).T
        norm_P3d_roted_centered = np.linalg.norm(P3d_roted_centered[0:2, :])
        norm_P2d_centered = np.linalg.norm(P2d_centered)
        P3d_roted_centered_resized = P3d_roted_centered *\
            norm_P2d_centered/norm_P3d_roted_centered
        skel = np.vstack((P2d_croped, P3d_roted_centered_resized[2, :]))
        return skel

    def getSkel_Label(self, index):
        skel = self.getSkel(index)
        _, label = self.getImgName_onehotLabel(index)
        return skel, label

    def getSkel_Label_all(self, num):
        from tqdm import tqdm
        l = random.sample(range(self.nSamples), num)
        skels = np.zeros((num, 3, ref.nJoints))
        labels = np.zeros((num, len(ref.actions)))
        for i, index in tqdm(enumerate(l)):
            skels[i], labels[i] = self.getSkel_Label(index)
        return skels, labels

    def getRotation(self, index):
        camR = self.annot['cam'][0][0][index][0][0][0]['R']
        return camR

    def getSize(self):
        return self.nSamples

    def getAll(self, num, replace=False):
        from tqdm import tqdm
        self.cache_base_path = globalConfig.cache_base_path
        pickleCachePath = '{}/h36m_{}_{}.pkl'.format(self.cache_base_path,
                                                     self.split, str(num))
        imgs = np.zeros((num, 128, 128, 1))
        labels = np.zeros(num)
        if os.path.isfile(pickleCachePath) and not replace:
            print('direct load from the cache')
            t1 = time.time()
            f = open(pickleCachePath, 'rb')

            # (self.frmList) += pickle.load(f)
            data = pickle.load(f)

            t1 = time.time() - t1
            print('loaded with {}s'.format(t1))
            print(data[1].shape)
            return data[0].astype(np.int), data[1].astype(np.int)

        l = random.sample(range(self.nSamples), num)

        for i, index in tqdm(enumerate(l)):
            imgname, label = self.getImgName_Label(index)
            img = cv2.imread(imgname, 0)
            if img is None:
                print(imgname)
                break
            img = img.reshape(1, 128, 128, 1)
            imgs[i] = img
            labels[i] = label

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        pickle.dump((imgs, labels), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('loaded with {} frames'.format(num))

        return imgs.astype(np.int), labels.astype(np.int)


# a = H36M('valid')
#print(a.getRotation(1))
#print(a.getPart3D(1))
# print(a.getSkel(1412))
