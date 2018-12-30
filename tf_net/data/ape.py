import sys
sys.path.append('../data/')
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import time
import os
from pprint import pprint
import random
import numpy as np
import ref
import globalConfig

class APE:
    def __init__(self):
        # print('==>initializing 3D data.')
        activityLabelPath = {}

        activityLabelPath = os.path.join(globalConfig.ape_base_path, 'activityLabel'+'.txt')
        self.folder=[]
        with open(activityLabelPath,'r') as f:
            action=[]
            labelfile = f.readlines()
            for i in range(len(labelfile)-1):
                line = labelfile[i]
                self.folder.append(line[:6])
                action.append(line[7:-1])
            values = np.array(action)

            label_encoder = LabelEncoder()
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = label_encoder.fit_transform(values)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            self.action = onehot_encoder.fit_transform(integer_encoded)

        self.nFolder = len(labelfile)-1

    def readallData(self):
        images = []
        depths = []
        skels = []
        labels = []
        for i in range(self.nFolder):
            image, depth, skel = self.readData(i)
            for j in range(len(skel)):
                labels.append(self.action[i])
                images.append(image[i])
                depths.append(depth[i])
                skels.append(skel[i])

        self.nsamples = len(skels)
        return images, depths, skels, labels

    def readData(self, folderID):
        image = self.readImage(folderID)
        depth = self.readDepth(folderID)
        skel = self.readSkel(folderID)
        return image, depth, skel
        
    def readImage(self, folderID):
        images = []
        folderpath = os.path.join(
            globalConfig.ape_base_path, self.folder[folderID])
        numofImages = len([val for val in os.listdir(folderpath) if 'image' in val])

        for i in range(numofImages):
            imagename = 'image{:05d}.jpg'.format(i)
            imagename = os.path.join(folderpath,imagename)
            images.append(imagename)

        return np.asarray(images)

    def readDepth(self, folderID):
        depths = []
        folderpath = os.path.join(
            globalConfig.ape_base_path, self.folder[folderID])
        numofImages = len([val for val in os.listdir(folderpath) if 'depth' in val])

        for i in range(numofImages):
            depthname = 'image{:05d}.jpg'.format(i)
            depthname = os.path.join(folderpath,depthname)
            depths.append(depthname)

        return np.asarray(depths)

    def readSkel(self, folderID):
        skels = []
        Skelfilepath = os.path.join(
            globalConfig.ape_base_path, self.folder[folderID]+'.txt')
        with open(Skelfilepath, 'r') as f:
            next(f)
            for line in f:
                skel = []
                fields = line.split(" ")
                for i in range(2, len(fields)-1, 14):
                    skel.append(fields[i])
                for i in range(3, len(fields)-1, 14):
                    skel.append(fields[i])
                for i in range(4, len(fields)-1, 14):
                    skel.append(fields[i])
                skels.append(skel)
        return np.asarray(skels)

# a=APE()
