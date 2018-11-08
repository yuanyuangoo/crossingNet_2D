import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]
edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
         [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


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


def drawskelCV(skel):
    if not skel.shape == (3, 17):
        skel = np.reshape(skel, (3, 17))
    skel=skel.T
    axis = [1, 1, 0]
    theta = 45

    img = 255*np.ones((128, 128))
    for edge in edges:
        pt1 = 128*skel[edge[0]]
        pt2 = 128*skel[edge[1]]

        cv2.line(img, (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])), (0, 255, 0), 4)
    cv2.imwrite("im.jpg", img)


s1 = np.array([0.08320184, 0.07264254, 0.0706107, 0.06962919, 0.09361494, 0.09546421, 0.10024675, 0.0841208, 0.08456783, 0.08378701, 0.08583064, 0.06971817, 0.04401781, 0.02061944, 0.10017963, 0.12358516, 0.13991928, 0.07750564, 0.07880584, 0.12088734, 0.16266072, 0.07624983, 0.11768702, 0.15843196, 0.05470486,
               0.03210447, 0.02203918, 0.01285243, 0.03362578, 0.0396279, 0.04090642, 0.03423049, 0.0404605, 0.04268303, 0., 0.00380243, 0.00109286, 0.00149847, 0.00380236, 0.00951302, 0.00962369, 0.00536631, 0.01094982, 0.00879497, - 0.01515482, 0.01908453, 0.02843186, 0.03015406, 0.00500554, 0.00633337, 0.02029877])
# drawskel(s1)
drawskelCV(s1)
