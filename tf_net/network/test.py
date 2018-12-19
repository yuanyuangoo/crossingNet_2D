from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.models import Model, Sequential
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Conv2DTranspose, Flatten, Reshape,LeakyReLU,BatchNormalization
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

resNet50 = ResNet50(weights='imagenet', include_top=False,
                    pooling=max, input_shape=(128, 128, 3))
last = resNet50.output
x = Flatten(input_shape=resNet50.output_shape[1:])(last)
x = BatchNormalization()(x)
x = Dense(2**10)(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Dense(2**14, activation='tanh')(x)

y_pred = Reshape([128, 128,1])(x)

for layer in resNet50.layers:
    layer.trainable = False

model = Model(resNet50.input, y_pred)
model.summary()
model.compile(loss='mean_absolute_error',
              optimizer='adam')
train_dataset = Dataset()
train_dataset.loadH36M(1024, mode='train', tApp=True, replace=False)

valid_dataset = Dataset()
valid_dataset.loadH36M(64, mode='valid', tApp=True, replace=False)

test_labels, test_skel, test_img, test_img_rgb, _, _ = prep_data(
    valid_dataset, 64)
train_labels, train_skel, train_img, train_img_rgb, _, batch_idxs = prep_data(
    train_dataset, 64)
sample_dir = globalConfig.p2pr_pretrain_path
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

for epoch in range(1000):
    cost = model.train_on_batch(
        train_img_rgb, train_img, sample_weight=None, class_weight=None)
    print('epoch: {}, loss: {}'.format(epoch, cost))
    if epoch % 10 ==1:
        test = model.predict(test_img_rgb[0:2])
        cv2.imwrite(sample_dir + "test_{}.jpg".format(epoch),
                    (test[0]+1)*255.0/2)



model.save('backgroundsub.h5')
