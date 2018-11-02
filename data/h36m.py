import os
from pprint import pprint
import globalConfig
import random
from util import Rnd, Flip, ShuffleLR
import cv2
import ref
import numpy as np
import sys
sys.path.append('./')
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(list(ref.tags.keys()))
array = enc.transform("Directions").toarray()
print(array)
class H36M:
    def __init__(self, split):
        print('==>initializing3Ddata.')
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
        #ram=np.asarray(range(self.nSamples))
        #random.shuffle(ram)

        #self.id=ram
        self.root = 1
        #self.opt=opt
        self.split = split
        self.annot = annot

        print('Loaded 3D {} samples'.format(len(self.annot['scale'])))

    def GetPart3D(self, index):
        #pts_3d_mono=self.annot['S'][index].copy()
        pts_3d = self.annot['S_glob'][0][0][index].copy()
        #pts_3d=pts_3d-pts_3d[self.root]
        #L2_Norm=np.sum(np.abs(pts_3d)**2,axis=-1)**(1./2)
        #return (pts_3d/max(L2_Norm)+1)*64
        return pts_3d

    def GetPart2D(self, index):
        pts_2d = self.annot['part'][0][0][index].copy()
        return pts_2d

    def GetImgName_Label(self, index):
        #index=self.id[index]
        imgname = ref.h36mImgDir + \
            self.annot['imgname'][0][0][index][0][0][:]+'.jpg'
        label = self.GetLabel(imgname)
        return imgname, label

    def GetLabel(self, imgname):

        for key, tag in tags.items():
            if tag in imgname:
                return key
        return "Walking"

    def GetSkel(self, index):
        #index=self.id[id]
        R = self.GetRotation(index)
        P2d = self.GetPart2D(index)
        P2d_centered = P2d-P2d[ref.root]
        P3d = self.GetPart3D(index)

        P3d_roted = np.dot(R, P3d)
        P3d_roted_centered = (P3d_roted.T-(P3d_roted[:, 1])).T
        norm_P3d_roted_centered = np.linalg.norm(P3d_roted_centered[0:2, :])
        norm_P2d_centered = np.linalg.norm(P2d_centered)
        P3d_roted_centered_resized = P3d_roted_centered *\
            norm_P2d_centered/norm_P3d_roted_centered
        skel = np.vstack((P2d, P3d_roted_centered_resized[2, :]))
        return skel

    def GetRotation(self, index):
        camR = self.annot['cam'][0][0][index][0][0][0]['R']
        return camR

    def GetSize(self):
        return self.nSamples


#a=H36M('valid')
#print(a.GetRotation(1))
#print(a.GetPart3D(1))
