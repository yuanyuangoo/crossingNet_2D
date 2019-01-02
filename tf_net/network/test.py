import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance=21):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

def SkelGaussianHeatMap(img_height, img_width, skel):
    n_joints = int(len(skel)/3)
    pose_heatmap = np.zeros((img_height, img_width, n_joints))
    z_heatmap = np.zeros((img_height, img_width, n_joints))
    skel = np.reshape(skel, (3, n_joints))
    for idx in range(n_joints):
        joint = skel[:, idx]
        pose_heatmap[:, :, idx] = CenterGaussianHeatMap(
            img_height, img_width, joint[0], joint[1])
        z_heatmap[:, :, idx] = joint[2]*np.ones((img_height, img_width))
    return pose_heatmap, z_heatmap


image_file = '1.jpg'
img = cv2.imread(image_file)
img = img[:, :, ::-1]

height, width,_ = np.shape(img)
cy, cx = height/2.0, width/2.0

start = time.time()
heatmap1 = CenterLabelHeatMap(width, height, cx, cy, 21)
t1 = time.time() - start

start = time.time()
heatmap2 = CenterGaussianHeatMap(height, width, cx, cy, 21)
t2 = time.time() - start

skel = np.repeat(cy, 51)
# skel = np.reshape(skel, (3, 17))
heatmap,z_map = SkelGaussianHeatMap(height, width, skel)
maxindex = heatmap[:, :, 0].argmax()
index=np.unravel_index(maxindex, (128,128))
print(index)
print(t1, t2)

plt.subplot(1,2,1)
plt.imshow(heatmap[:,:,0])
plt.subplot(1,2,2)
plt.imshow(z_map[:,:,0])
plt.show()

print('End.')
