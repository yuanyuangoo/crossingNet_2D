import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]
edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
         [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


def drawskel(skel):
    if not skel.shape == (17, 3):
        np.reshape(skel, (17, 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    for point in skel:
        point[2] = point[2]
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
        # ys.append(1)
    ax.scatter(xs, ys, zs)

    for edge in edges:
        pt1 = skel[edge[0]]
        pt2 = skel[edge[1]]
        x = [pt1[0], pt2[0]]
        y = [pt1[1], pt2[1]]
        z = [pt1[2], pt2[2]]
        # y = [1, 1]
        ax.plot(x, y, z)
    maxC = max(max(xs), max(ys), max(zs))
    minC = min(min(xs), min(ys), min(zs))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(minC, maxC)
    ax.set_ylim(minC, maxC)
    ax.set_zlim(minC, maxC)
    plt.show()


s1 = np.array([[0.,    0.,    0.],
               [-48.391846,   37.981995, -123.73633],
               [-227.9494,  451.45993,  -22.82129],
               [-197.43384,  887.92413,  119.91553],
               [48.419678,  -37.789734,  123.78369],
               [29.26593,  420.4163,  179.18555],
               [127.46741,  869.75775,  163.23193],
               [31.023132, -243.06036,  -67.96094],
               [-16.318604, -487.08453,  -94.12744],
               [-108.2832, -543.1556,  -66.9541],
               [-112.63605, -657.65607,  -76.72412],
               [-27.38971, -431.00104, -250.32568],
               [76.55267, -223.64935, -412.76904],
               [-4.810669,  -36.930298, -270.73193],
               [4.9660034, -437.17255,   63.104004],
               [-16.765259, -206.76227,  226.2749],
               [-161.73285,  -32.317505,  327.3374]], dtype=np.float32)
drawskel(s1)
