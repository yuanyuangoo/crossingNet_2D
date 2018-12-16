from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('./')
from data.layers import *
import globalConfig
import data.plot_utils as plot_utils
from data.util import *
from data.dataset import *
import data.ref as ref

model = ResNet50(weights='imagenet', include_top=False)
train_dataset = Dataset()
train_dataset.loadH36M(1024, mode='valid', tApp=True, replace=False)
train_labels, train_skel, train_img, train_img_rgb, _, batch_idxs = prep_data(
    train_dataset, 64)

img_data = preprocess_input(train_img_rgb)

resnet50_feature = model.predict(img_data)

print(resnet50_feature.shape)
