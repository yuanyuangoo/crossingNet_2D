import tensorflow.contrib.slim as slim
from collections import namedtuple
from numpy.matlib import repmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
# import globalConfig
from numpy.random import randn
import math
# import torch
import scipy.misc
from six.moves import xrange
import data.ref as ref
import tensorflow as tf
# CameraOption = namedtuple('CameraOption', [
#                           'focal_x', 'focal_y', 'center_x', 'center_y', 'width', 'height', 'far_point'])
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


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


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

    def __init__(self, img=None, skel=None, label=None, path=None):
        # if not isinstance(com2D, np.ndarray):
        #     (self.crop_dm, self.trans, self.com3D) = dm.Detector()
        # else:
        #     (self.crop_dm, self.trans, self.com3D) = dm.cropArea3D(dm.dmData, com2D)
        self.norm_img = img.Data
        if isinstance(skel, np.ndarray):
            if len(skel) % 3 != 0:
                raise ValueError('invalid length of the skeleton mat')
            # jntNum = len(skel)/3
            self.label = label
            self.with_skel = True
            self.skel = skel.astype(np.float32)
            self.path=path
            # self.norm_skel = skel.astype(np.float32)
            #crop_skel is the training label for neurual network, normalize wrt com3D
            # self.crop_skel = (self.skel - repmat(self.com3D, 1, jntNum))[0]
            # self.crop_skel = self.crop_skel.astype(np.float32)
            # self.normSkel()
        else:
            self.skel = None
            self.crop_skel = None
            self.with_skel = False
            self.label = label

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


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def drawImageCV(skel, axis=(0, 1, 0), theta=0):
        if not skel.shape == (3, 17):
            skel = np.reshape(skel, (3, 17))
        skel = skel.T
        skel = np.dot(skel, rotation_matrix(axis, theta))
        skel = skel[:, 0:2]
        min_s = skel.min()
        max_s = skel.max()
        mid_s = (min_s+max_s)/2
        skel = (((skel-mid_s)/(max_s-min_s))+0.52)*125

        img = 255*np.ones((128, 128, 3))
        for i, edge in enumerate(ref.edges):
            pt1 = skel[edge[0]]
            pt2 = skel[edge[1]]

            cv2.line(img, (int(pt1[0]), int(pt1[1])),
                     (int(pt2[0]), int(pt2[1])), (np.asarray(ref.color[i])*255).tolist(), 4)
        return img


def inverse_transform(images):
    return (images+1)*255.0/2


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    # return scipy.misc.imsave(path, image)
    return cv2.imwrite(path, image)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(
            image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')
