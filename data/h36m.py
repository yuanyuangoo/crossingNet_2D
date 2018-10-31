# import torch.utils.data as data
import numpy as np
import ref
# import torch
import h5py
import cv2
from util import Rnd, Flip, ShuffleLR
from img import Crop, DrawGaussian, Transform3D
# import pprint as pp
import random


class H36M:
  def __init__(self, split):
    print ('==> initializing 3D data.')
    annot = {}

    tags = ['S',     'center',  'index',     'normalize',     'part',
            'person',     'scale',     'torsoangle',     'visible',     'zind']
    f = h5py.File('/media/a/D/datasets/h36m/annot/'+split+'.h5', 'r')
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    annot['imgname'] = {}
    self.nSamples = len(annot['scale'])

    with open('/media/a/D/datasets/h36m/annot/'+split+'_images.txt', 'r') as f:
      for i in range(self.nSamples):
        annot['imgname'][i] = f.readline()[:-1]

    ram = np.asarray(range(self.nSamples))
    random.shuffle(ram)

    self.id = ram
    self.root = 1
    # self.opt = opt
    self.split = split
    self.annot = annot

    print ('Loaded 3D {} samples'.format(len(self.annot['scale'])))

  def LoadImage(self, index):
    id = self.id[index]
    path = ref.h36mImgDir+self.annot['imgname'][id]
    img = cv2.imread(path)
    # while(1):
    #   cv2.imshow(path, img)
    #   c = cv2.waitKey(1)
    #   if c == 27:
    #     break
    return img

  def GetPart(self, index):
    index = self.id[index]
    # pts_3d_mono = self.annot['S'][index].copy()
    pts_3d = self.annot['S'][index].copy()
    # pts_3d = pts_3d - pts_3d[self.root]
    # L2_Norm = np.sum(np.abs(pts_3d)**2, axis=-1)**(1./2)
    # return (pts_3d/max(L2_Norm)+1)*64
    return pts_3d

  def getImgName(self, index):
    index = self.id[index]
    return ref.h36mImgDir+self.annot['imgname'][index]

  def getSize(self):
    return self.nSamples
  # def GetPartInfo(self, index):
  #   pts = self.annot['part'][index].copy()
  #   pts_3d_mono = self.annot['S'][index].copy()
  #   pts_3d = self.annot['S'][index].copy()
  #   c = np.ones(2) * ref.h36mImgSize / 2
  #   s = ref.h36mImgSize * 1.0

  #   pts_3d = pts_3d - pts_3d[self.root]

  #   s2d, s3d = 0, 0
  #   for e in ref.edges:
  #     s2d += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
  #     s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
  #   scale = s2d / s3d

  #   for j in range(ref.nJoints):
  #     pts_3d[j, 0] = pts_3d[j, 0] * scale + pts[self.root, 0]
  #     pts_3d[j, 1] = pts_3d[j, 1] * scale + pts[self.root, 1]
  #     pts_3d[j, 2] = pts_3d[j, 2] * scale + ref.h36mImgSize / 2
  #   return pts, c, s, pts_3d, pts_3d_mono

  # def getitem(self, index):
  #   if self.split == 'train':
  #     index = np.random.randint(self.nSamples)
  #   img = self.LoadImage(index)
  #   pts, c, s, pts_3d, pts_3d_mono = self.GetPartInfo(index)
  #   pts_3d[7] = (pts_3d[12] + pts_3d[13]) / 2

  #   inp = Crop(img, c, s, 0, ref.inputRes) / 256.
  #   outMap = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
  #   outReg = np.zeros((ref.nJoints, 3))
  #   for i in range(ref.nJoints):
  #     pt = Transform3D(pts_3d[i], c, s, 0, ref.outputRes)
  #     if pts[i][0] > 1:
  #       outMap[i] = DrawGaussian(outMap[i], pt[:2], ref.hmGauss)
  #     outReg[i, 2] = pt[2] / ref.outputRes * 2 - 1
  #   print(inp.shape)
  #   # inp = torch.from_numpy(inp)
  #   return inp, outMap, outReg, pts_3d_mono

  # def __len__(self):
  #   return self.nSamples
# a = H36M('train')
# print(a.getImgName(1))
# print(a.GetPart(1))