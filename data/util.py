from collections import namedtuple
from numpy.matlib import repmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import cv2
# import globalConfig
from numpy.random import randn
import ref
# import torch

CameraOption = namedtuple('CameraOption', [
                          'focal_x', 'focal_y', 'center_x', 'center_y', 'width', 'height', 'far_point'])
# Frame = namedtuple('Frame', ['dm', 'skel', 'crop_dm', 'crop_skel', 'file_name'])
# Frame.__new__.__defaults__ = (None,)*len(Frame._fields)

figColor = [(19, 69, 139),
            (51, 51, 255),
            (51, 151, 255),
            (51, 255, 151),
            (255, 255, 51),
            (255, 51, 153),
            (0, 255, 0)
            ]
nyuColorIdx = [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6 + [0]*6
nyuFigColorIdx = [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6
# for l in globalConfig.nyuKeepList:
# nyuColorIdx[l] = 6
iclColorIdx = [0] + [1]*3 + [2]*3 + [3]*3 + [4]*3 + [5]*3
msraColorIdx = [0] + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4


def initFigBone(startIdx, jntNum, color): return \
    [(sIdx, eIdx, color) for sIdx, eIdx in
     zip(range(startIdx, startIdx+jntNum-1), range(startIdx+1, startIdx+jntNum))]


def flattenBones(bones):
    b = []
    for bb in bones:
        b += bb
    return b


nyuBones = flattenBones([initFigBone(b*6, 6, figColor[b+1]) for b in range(5)])
icvlBones = flattenBones(
    [initFigBone(b*3+1, 3, figColor[b+1]) for b in range(5)])
msraBones = flattenBones(
    [initFigBone(b*4+1, 4, figColor[b+1]) for b in range(5)])


class Frame(object):
    skel_norm_ratio = 50.0

    def __init__(self, img=None, skel=None, com2D=None, flag=None):
        # if not isinstance(com2D, np.ndarray):
        #     (self.crop_dm, self.trans, self.com3D) = dm.Detector()
        # else:
        #     (self.crop_dm, self.trans, self.com3D) = dm.cropArea3D(dm.dmData, com2D)
        self.norm_img = img.Data
        if isinstance(skel, np.ndarray):
            if len(skel) % 3 != 0:
                raise ValueError('invalid length of the skeleton mat')
            jntNum = len(skel)/3
            self.with_skel = True
            self.skel = skel.astype(np.float32)
            self.norm_skel = skel.astype(np.float32)
            #crop_skel is the training label for neurual network, normalize wrt com3D
            # self.crop_skel = (self.skel - repmat(self.com3D, 1, jntNum))[0]
            # self.crop_skel = self.crop_skel.astype(np.float32)
            # self.normSkel()
        else:
            self.skel = None
            self.crop_skel = None
            self.with_skel = False

    # save only the norm_dm and norm_skel for training, clear all initial size data

    def saveOnlyForTrain(self):
        self.img = None
        self.crop_img = None
        self.skel = None
        self.crop_skel = None
        # self.trans = None
        # self.com3D = None

    def normSkel(self):
        self.norm_skel = self.crop_skel.copy() / self.skel_norm_ratio

    # inverse process, input: norm_skel, output: crop_skel, skel
    def setNormSkel(self, norm_skel):
        if len(norm_skel) % 3 != 0:
            raise ValueError('invalid length of the skeleton mat')
        jntNum = len(norm_skel)/3
        self.norm_skel = norm_skel.copy().astype(np.float32)
        self.crop_skel = norm_skel.copy()*self.skel_norm_ratio
        self.com3D = np.zeros([3])
        self.com3D[2] = 200
        self.skel = (self.crop_skel + repmat(self.com3D, 1, jntNum))[0]
        self.skel = self.skel.astype(np.float32)

    def setCropSkel(self, crop_skel):
        if len(crop_skel) % 3 != 0:
            raise ValueError('invalid length of the skeleton mat')
        jntNum = len(crop_skel)/3
        self.crop_skel = crop_skel.astype(np.float32)
        self.skel = (self.crop_skel + repmat(self.com3D, 1, jntNum))[0]
        self.skel = self.skel.astype(np.float32)
        self.normSkel()

    def setSkel(self, skel):
        if len(skel) % 3 != 0:
            raise ValueError('invalid length of the skeleton mat')
        jntNum = len(skel)/3
        self.skel = skel.astype(np.float32)
        #crop_skel is the training label for neurual network, normalize wrt com3D
        self.crop_skel = (self.skel - repmat(self.com3D, 1, jntNum))[0]
        self.crop_skel = self.crop_skel.astype(np.float32)
        self.normSkel()

    def saveAnnotatedSample(self, path):
        skel2 = self.crop2D()
        skel2 = skel2.reshape(-1, 3)
        for i, pt in enumerate(skel2):
            skel2[i] = Camera.to2D(pt)
        print ('current camera option={}'.format(Camera.focal_x))

        skel = self.skel
        skel.shape = (-1, 3)

        dm = self.norm_dm.copy()
        dm[dm == Camera.far_point] = 0
        ax = fig.add_subplot(121)
        img = dm.copy()
        img = img - img.min()
        img *= 255/img.max()
        for pt in skel2:
            cv2.circle(img, (pt[0], pt[1]), 2, (255, 0, 0), -1)
        cv2.imwrite(path, img)

    def showAnnotatedSample(self):
        fig = plt.figure()
        fig.suptitle('annoated example')
        if self.dm is not None and self.skel is not None:
            skel2 = self.skel.copy()
            skel2 = skel2.reshape(-1, 3)
            for i, pt in enumerate(skel2):
                skel2[i] = Camera.to2D(pt)

            print ('current camera option={}'.format(Camera.focal_x))

            skel = self.skel
            skel.shape = (-1, 3)

            dm = self.dm.copy()
            dm[dm == Camera.far_point] = 0
            ax = fig.add_subplot(121)
            ax.imshow(dm, cmap=matplotlib.cm.jet)
            ax.scatter(skel2[:, 0], skel2[:, 1], c='r')
            ax.set_title('initial')

        if self.crop_dm is not None and self.crop_skel is not None:
            skel2 = self.crop2D()

            ax = fig.add_subplot(122)
            ax.imshow(self.crop_dm, cmap=matplotlib.cm.jet)
            ax.scatter(skel2[:, 0], skel2[:, 1], c='r')
            ax.set_title('cropped')

        if self.norm_dm is not None and self.norm_skel is not None:
            skel2 = self.crop2D()
            ax = fig.add_subplot(122)
            ax.imshow(self.norm_dm, cmap=matplotlib.cm.jet)
            ax.scatter(skel2[:, 0], skel2[:, 1], c='r')
            ax.set_title('normed')
        plt.show(block=False)

    def crop2D(self):
        self.crop_skel = self.norm_skel * np.float32(50.0)
        skel = self.crop_skel.copy()
        jntNum = len(skel)/3
        skel = skel.reshape(-1, 3)
        skel += repmat(self.com3D, jntNum, 1)
        for i, jnt in enumerate(skel):
            jnt = Camera.to2D(jnt)
            pt = np.array([jnt[0], jnt[1], 1.0], np.float32).reshape(3, 1)
            pt = self.trans*pt
            skel[i, 0], skel[i, 1] = pt[0], pt[1]
        return skel

    def full2D(self):
        '''
        2D transformation of the cropped_pt(the estimated one) to the initial
        size image
        '''
        skel = self.skel.copy()
        skel = skel.reshape(-1, 3)

        for i, jnt in enumerate(skel):
            jnt = Camera.to2D(jnt)
            skel[i, 0], skel[i, 1] = jnt[0], jnt[1]
        return skel

    def visualizeCrop(self, norm_skel=None):
        img = self.norm_dm.copy()
        img = (img+0.5)*255.0
        colorImg = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if norm_skel is None:
            return colorImg

        self.setNormSkel(norm_skel)
        skel2D = self.crop2D()
        for pt in skel2D:
            cv2.circle(colorImg, (pt[0], pt[1]), 2, (0, 0, 255), -1)
        return colorImg

    def visualizeFull(self, norm_skel=None):
        img = self.dm.copy()
        img[img >= Camera.far_point] = 0
        img = img*(256/img.max())
        colorImg = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if norm_skel is None:
            return colorImg

        self.setNormSkel(norm_skel)
        skel2D = self.full2D()
        for pt in skel2D:
            cv2.circle(colorImg, (pt[0], pt[1]), 5, (0, 0, 255), -1)
        return colorImg


def vis_pose(normed_vec):

    vec = normed_vec.copy()
    vec = normed_vec.copy()*50.0

    vec.shape = (-1, 3)

    img = np.ones((128, 128))*225
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)

    for idx, pt3 in enumerate(vec):
        # pt = Camera.to2D(pt3)
        # pt = pt3[0:2]
        # pt = (pt[0], pt[2])
        cv2.circle(img, (int(pt3[0]), int(pt3[2])), 2, (255, 0, 0), -1)
    return img


def vis_normed_pose(normed_vec, img=None):
    import depth
    pt2 = projectNormPose3D(normed_vec)

    if not type(img) is np.ndarray:
        img = np.ones((depth.DepthMap.size2[0], depth.DepthMap.size2[1]))*255

    img = img.reshape(depth.DepthMap.size2[0], depth.DepthMap.size2[1])
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    for idx, pt in enumerate(pt2):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    return img


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))


def Flip(img):
  return img[:, :, ::-1].copy()


def ShuffleLR(x):
  for e in ref.shuffleRef:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x
