import time
import sys
from keras.layers import Flatten, Dense, Dropout, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('./')
import globalConfig
from data.dataset import *
from data.util import *

def vnect(train_dataset, valid_dataset):
    epochs = 10
    batch_size=64
    _, train_skel, _, train_img_rgb, n_samples, total_batch = prep_data(
        train_dataset, batch_size)
    _, test_skel, _, test_img_rgb, _, _ = prep_data(
        valid_dataset, batch_size)
    print("Preparing heatmap!")
    input_size = 224
    output_size=224
    n_joints=17
    train_heat_maps = np.zeros(
        (n_samples, input_size, input_size, n_joints))
    train_z_heat_maps = np.zeros(
        (n_samples, input_size, input_size, n_joints))
    for idx in tqdm(range(n_samples)):
        train_heat_maps[idx], train_z_heat_maps[idx] = SkelGaussianHeatMap(
            input_size, output_size, train_skel[idx]*256-128)

    print("Prepare heatmap completed!")
    #load ResNet50 without dense layer and with theano dim ordering
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))

    res5a = base_model.get_layer('activation_43')
    res4d = base_model.get_layer('activation_34')

    all_layers = base_model.layers
    for i in range(base_model.layers.index(res4d)):
        all_layers[i].trainable = False

    outputs = []

    res4d_heatmap1a = Conv2DTranspose(17, 4, strides=2, padding='same',
                                      name='res4d_heatmap1a')(res4d.output)
    res4d_heatmap_bn = BatchNormalization(
        name='res4d_heatmap_bn')(res4d_heatmap1a)
    outputs.append(res4d_heatmap_bn)

    res5a_heatmap1a = Conv2DTranspose(17, 4, strides=4, padding='same',
                                      name='res5a_heatmap1a')(res5a.output)
    res5a_heatmap_bn = BatchNormalization(
        name='res5a_heatmap_bn')(res5a_heatmap1a)
    outputs.append(res5a_heatmap_bn)

    #create graph of your new model
    head_model = Model(input=base_model.input, output=outputs)

    #compile the model
    head_model.compile(optimizer='rmsprop',
                       loss='mean_squared_error', metrics=['accuracy'])

    head_model.fit(x=train_img_rgb, y=train_heat_maps, batch_size=64, epochs=epochs, verbose=1, callbacks=None, validation_split=0.0, validation_data=None,
                   shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

    return 1
if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        # ds.loadH36M_expended(64*10, mode='train',
        #                      tApp=True, replace=False)
        ds.loadH36M(64*10, mode='train',
                    tApp=True, replace=False)
        val_ds = Dataset()
        # val_ds.loadH36M_all('all', mode='valid',
        #                     tApp=True, replace=False)
        val_ds.loadH36M(64, mode='valid',
                        tApp=True, replace=False)
        Vnect = vnect(ds, val_ds)

    elif globalConfig.dataset == 'APE':
        ds = Dataset()
        ds.loadApe(64*300, mode='train', tApp=True, replace=False)

        val_ds = Dataset()
        val_ds.loadApe(64, mode='valid', tApp=True, replace=False)

        Vnect = vnect()

    else:
        raise ValueError('unknown dataset %s' % globalConfig.dataset)
