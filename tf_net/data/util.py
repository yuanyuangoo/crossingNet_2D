from numpy.matlib import repmat
import numpy as np
import os
import cv2
from numpy.random import randn
import math
# import torch
import scipy.misc
from six.moves import xrange
import data.ref as ref


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

    def __init__(self, img=None, img_RGB=None, skel=None, label=None, path=None):
        self.norm_img_RGB = img_RGB.Data
        if isinstance(skel, np.ndarray):
            if len(skel) % 3 != 0:
                raise ValueError('invalid length of the skeleton mat')
            # jntNum = len(skel)/3
            self.label = label
            self.with_skel = True
            self.norm_skel = skel

            self.path = path

        else:
            self.skel = None
            self.crop_skel = None
            self.with_skel = False
            self.label = label
            self.norm_img = img.Data

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


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        batch, row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (batch, row, col, ch))
        gauss = gauss.reshape(batch, row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        batch, row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = -1
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        batch, row, col, ch = image.shape
        gauss = np.random.randn(batch, row, col, ch)
        gauss = gauss.reshape(batch, row, col, ch)
        noisy = image + image * gauss
        return noisy


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


def drawImageCV(skel, img=None, axis=(0, 1, 0), theta=0):
    numofJoint = 17
    if not skel.shape == (3, numofJoint):
        skel = np.reshape(skel, (3, numofJoint))
    skel = skel.T
    skel = np.dot(skel, rotation_matrix(axis, theta))
    skel = skel[:, 0:2]
    skel = (skel*256)-128
    # min_s = skel.min()
    # max_s = skel.max()
    # mid_s = (min_s+max_s)/2
    # skel = (((skel-mid_s)/(max_s-min_s))+0.52)*125
    if img is None:
        img = 255*np.ones((128, 128, 3))
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i, edge in enumerate(ref.h36medges):
        pt1 = skel[edge[0]]
        pt2 = skel[edge[1]]

        cv2.line(img, (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])), (np.asarray(ref.color[i])*255).tolist(), 4)
    return img


def inverse_transform(images):
    return (images+1)*255.0/2


def imsave(images, size, path, skel=None):
    image = np.squeeze(merge(images, size, skel))
    # return scipy.misc.imsave(path, image)
    return cv2.imwrite(path, image)


def save_images(images,  size, image_path, skel=None):
    return imsave(inverse_transform(images), size, image_path, skel)


def save_images_one_by_one(images, path, idx):
    if not os.path.exists(path):
        os.makedirs(path)
    for imid in range(images.shape[0]):
        img = inverse_transform(images[imid])
        img_save_name = '{}/predict_{:04d}.png'.format(path, idx*64+imid)
        cv2.imwrite(img_save_name, img)

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


def merge(images, size, skel=None):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            if skel is not None:
                image = drawImageCV(skel[idx], image)
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        c = 3
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            if skel is not None:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = drawImageCV(skel[idx], image)
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def prep_data(dataset, batch_size, skel=True, with_background=False):

    labels = []
    img = []
    img_RGB = []
    for frm in dataset.frmList:
        labels.append(frm.label)
        img_RGB.append(frm.norm_img_RGB)

    labels = np.asarray(labels)
    img_RGB = np.asarray(img_RGB)

    n_samples = img_RGB.shape[0]
    total_batch = int(n_samples / batch_size)

    seed = 547
    np.random.seed(seed)
    idx = np.array(range(n_samples))
    np.random.shuffle(idx)

    labels = labels[idx]
    img_RGB = img_RGB[idx]

    if skel:
        skel = []
        for frm in dataset.frmList:
            skel.append(frm.norm_skel)
        skel = (np.asarray(skel)+128)/256
        skel = np.asarray(skel)
        skel = skel[idx]
    else:
        skel = 1

    if with_background:
        for frm in dataset.frmList:
            img.append(frm.norm_img)
        img = np.asarray(img)
        img = img[idx]

    return labels, skel, img, img_RGB, n_samples, total_batch


def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance=21):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[x_p, y_p] = np.exp(-exponent)
    # gaussian_map = cv2.resize(
    #     gaussian_map, (img_height//8, img_width//8), interpolation=cv2.INTER_LINEAR)
    return gaussian_map


def calculateAccuracy(result, ground_truth):
    a = 1

origin = CenterGaussianHeatMap(128*2, 128*2, 128, 128)
def getoneHeatmap(img_height, img_width, c_x, c_y, variance=21):
    gaussian_map = origin[int(128-c_x):int(256-c_x), int(128-c_y):int(256-c_y)]
    gaussian_map = cv2.resize(gaussian_map, (img_height//8, img_width//8), interpolation=cv2.INTER_LINEAR)
    return gaussian_map
    
def SkelGaussianHeatMap(img_height, img_width, skel):
    n_joints = int(len(skel)/3)
    pose_heatmap = np.zeros((img_height//8, img_width//8, n_joints))
    z_heatmap = np.zeros((img_height//8, img_width//8, n_joints))
    skel = np.reshape(skel, (3, n_joints))
    for idx in range(n_joints):
        joint = skel[:, idx]
        pose_heatmap[:, :, idx] = getoneHeatmap(
            img_height, img_width, joint[0], joint[1])
        z_heatmap[:, :, idx] = joint[2] * \
            np.ones((img_height//8, img_width//8))
    return pose_heatmap, z_heatmap/128


def SkelFromOnemap(heat_map, z_map):
    n_joints = heat_map.shape[2]
    skel = np.zeros(n_joints*3)

    s = extract_2d_joint_from_heatmap(heat_map, 128)
    z_map = cv2.resize(z_map, (128, 128), interpolation=cv2.INTER_LINEAR)

    for idx in range(n_joints):
        x = int(s[idx, 0])
        y = int(s[idx, 1])
        z = z_map[x, y, idx]
        skel[idx] = x
        skel[idx+n_joints] = y
        skel[idx+2*n_joints] = z*128

    return skel


def SkelFromHeatmap(heatmap, z_map):
    batch_size = heatmap.shape[0]
    skel = []
    for idx in range(batch_size):
        H = heatmap[idx]
        Z = z_map[idx]
        skel.append(SkelFromOnemap(H, Z))
    return np.asarray(skel)


def extract_2d_joint_from_heatmap(heatmap, input_size):
    heatmap_resized = cv2.resize(
        heatmap, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    joints_2d = np.zeros((17, 2))
    for joint_num in range(heatmap_resized.shape[2]):
        joint_coord = np.unravel_index(
            np.argmax(heatmap_resized[:, :, joint_num]), (input_size, input_size))
        joints_2d[joint_num, :] = joint_coord

    return joints_2d


def get_dist_pck(pred, gt):
    dist_ratio = np.zeros((pred.shape[0], pred.shape[2]))
    for imgidx in range(64):
        refDist = np.linalg.norm(gt[imgidx, :, 12]-gt[imgidx, :, 5])

        dist_ratio[imgidx, :] = np.sqrt(
            sum(
                np.square(
                    pred[imgidx, :, :]-gt[imgidx, :, :]
                ), 0)
        )/refDist

    return dist_ratio


def compute_pck(dist_ratio, label, threshold):
    pck = np.zeros((len(threshold),label.shape[1]+1, dist_ratio.shape[1]+1))
    for jidx in range(dist_ratio.shape[1]):
        for i_label in range(label.shape[1]):
            idx_of_label = np.where(label[:, i_label] == 1)[0]
            if len(idx_of_label) == 0:
                continue
            for i, t in enumerate(threshold):
                pck[i, i_label, jidx] = 100 * \
                    np.mean(dist_ratio[idx_of_label[0], jidx] <= t)
    for i_label in range(label.shape[1]):
        idx_of_label = np.where(label[:, i_label] == 1)[0]
        if len(idx_of_label) == 0:
                continue
        for i, t in enumerate(threshold):
                pck[i, i_label, -1] = 100 * \
                    np.mean(dist_ratio[idx_of_label[0], :] <= t)

    for jidx in range(dist_ratio.shape[1]):
        for i, t in enumerate(threshold):
            pck[i, -1, jidx] = 100 * \
                np.mean(dist_ratio[:, jidx] <= t)

    for i, t in enumerate(threshold):
        pck[i, -1, -1] = 100 * \
            np.mean(dist_ratio[:, :] <= t)
    return pck


def eval_pck(pred, gt, label):
    pred = np.reshape(pred, (pred.shape[0], 3, 17))
    gt = np.reshape(gt, (gt.shape[0], 3, 17))

    dist_3d = get_dist_pck(pred, gt)
    pck_all_3d = compute_pck(dist_3d, label, [0.5, 0.2])

    dist_2d = get_dist_pck(pred, gt)
    pck_all_2d = compute_pck(dist_2d, label, [0.5, 0.2])
    return pck_all_3d, pck_all_2d
