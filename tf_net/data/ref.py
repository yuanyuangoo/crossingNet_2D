import os
import sys
sys.path.append('./')
sys.path.append('data/')

import globalConfig
import numpy as np
nJoints = 17
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
shuffleRef = [[0, 5], [1, 4], [2, 3],
              [10, 15], [11, 14], [12, 13]]
# edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
#          [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
#          [6, 8], [8, 9]]
h36medges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
         [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
color = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5],
         [0, 0.5, 0.5], [0.5, 1, 0], [0.5, 0, 1], [0.5, 1, 1], [1, 0, 0.5]]

apeedges = [[0, 1], [1, 2], [2, 7], [7, 8], [8, 13], [2, 9], [
    9, 10], [10, 14], [1, 3], [3, 4], [4, 11], [1, 5], [5, 6], [6, 12]]

h36mImgSize = 128

actions = ["Directions",
           "Discussion",
           "Eating",
           "Activities",
           "Greeting",
           "Taking photo",
           "Posing",
           "Making purchases",
           "Smoking",
           "Waiting",
           "Sitting on chair",
           "Talking on the phone",
           "Walking dog",
           "Walking together",
           "Walking"]

oneHoted = np.eye(len(actions))
tags = {"Directions": "Directions",
        "Discussion": "Discussion",
        "Eating": "Eating",
        "Activities": "Down",
        "Greeting": "Greeting",
        "Taking photo": "Photo",
        "Posing": "Posing",
        "Making purchases": "Purcha",
        "Smoking": "Smoking",
        "Waiting": "Waiting",
        "Sitting on chair": "Sit",
        "Talking on the phone": "Phon",
        "Walking dog": "Dog",
        "Walking together": "Together"}

outputRes = 64
inputRes = 256
root = 0
eps = 1e-6

momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8

globalConfig
scale = 0.25
rotate = 30
hmGauss = 1
hmGaussInp = 20
shiftPX = 50
disturb = 10

dataDir = '../data'
mpiiImgDir = '/home/zxy/Datasets/mpii/images/'
h36mImgDir = os.path.join(globalConfig.h36m_base_path, 'resized/')
h36mImgDir_origin = os.path.join(globalConfig.h36m_base_path, 'images_resized/')

expDir = '../exp'

nThreads = 4
