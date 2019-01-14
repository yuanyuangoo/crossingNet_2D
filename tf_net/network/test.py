import time
import sys
from keras.layers import Flatten, Dense, Dropout, Conv2DTranspose, Reshape, Lambda, Multiply
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json
from keras.applications.resnet50 import ResNet50
from keras.activations import hard_sigmoid

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('./')
from data.util import *
from data.dataset import *
import globalConfig

input_size = 224
output_size = 28
batch_size = 64
epochs = 100
input_scalar=100
def getData(train_dataset, valid_dataset, batch_size):
    _, train_skel, _, train_img_rgb, n_samples, total_batch = prep_data(
        train_dataset, batch_size)
    _, test_skel, _, test_img_rgb, _, _ = prep_data(
        valid_dataset, batch_size)
    print("Preparing heatmap!")

    n_joints = 17
    train_heat_map = np.zeros(
        (n_samples, output_size, output_size, n_joints))
    test_heat_map = np.zeros(
        (batch_size, output_size, output_size, n_joints))
    # train_z_heat_maps = np.zeros(
    #     (n_samples, output_size, output_size, n_joints))
    for idx in tqdm(range(n_samples)):
        train_heat_map[idx], _ = SkelGaussianHeatMap(
            input_size, output_size, 224*(train_skel[idx]*2-1))

    for idx in tqdm(range(test_skel.shape[0])):
        test_heat_map[idx], _ = SkelGaussianHeatMap(
            input_size, output_size, 224*(test_skel[idx]*2-1))

    train_heat_maps = []
    train_heat_maps.append(train_heat_map)
    train_heat_maps.append(train_heat_map)
    print("Prepare heatmap completed!")
    return train_heat_maps, train_img_rgb, test_img_rgb, train_skel, test_skel, n_samples, total_batch


def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    head_model = model_from_json(loaded_model_json)
    # load weights into new model
    head_model.load_weights("model.h5")
    print("Loaded model from disk")
    return head_model


def build_model():
    #load ResNet50 without dense lainput_scalaryer and with theano dim ordering
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))

    res5a = base_model.get_layer('activation_43')
    res4d = base_model.get_layer('activation_34')
    L = base_model.get_layer('activation_25')

    all_layers = base_model.layers
    for i in range(base_model.layers.index(L)):
        all_layers[i].trainable = False

    outputs = []

    res4d_heatmap1a = Conv2DTranspose(
        17, 4, strides=2, padding='same', activation='sigmoid', name='res4d_heatmap1a')(res4d.output)
    res4d_heatmap_bn = BatchNormalization(
        name='res4d_heatmap_bn')(res4d_heatmap1a)
    sc_mult1 = Lambda(lambda x: x * input_scalar)(res4d_heatmap_bn)
    # sc_mult1=Multiply()([res4d_heatmap_bn,100])
    outputs.append(sc_mult1)

    res5a_heatmap1a = Conv2DTranspose(
        17, 4, strides=4, padding='same', activation='sigmoid', name='res5a_heatmap1a')(res5a.output)
    # res5a_heatmap_fc = Reshape((28, 28, 17))(Dense(
    #     28*28*17, activation='sigmoid')(Flatten()(res5a_heatmap1a)))

    res5a_heatmap_bn = BatchNormalization(
        name='res5a_heatmap_bn')(res5a_heatmap1a)
    sc_mult2 = Lambda(lambda x: x * input_scalar)(res5a_heatmap_bn)
    # sc_mult2 = Multiply()([res5a_heatmap_bn, 100])

    outputs.append(sc_mult2)
    
    #create graph of your new model
    head_model = Model(input=base_model.input, output=outputs)
    return head_model


def save_model(head_model):
    # serialize model to JSON
    model_json = head_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    head_model.save_weights("model.h5")
    print("Saved model to disk")

def train(head_model,train_img_rgb,train_heat_maps):
    #compile the model
    head_model.compile(optimizer='rmsprop',
                       loss='mean_absolute_error', metrics=['accuracy'])
    head_model.fit(x=train_img_rgb, y=train_heat_maps*100,
                   batch_size=batch_size, epochs=epochs)

    save_model(head_model)
    return head_model

def predict(head_model, test_img_rgb, total_batch):
    predict = head_model.predict(test_img_rgb)
    skel = SkelFromHeatmap(predict[1], predict[1], input_size)
    sample_dir = './'
    save_images(test_img_rgb, image_manifold_size(batch_size), '{}/test_{:02d}_{:04d}.png'.format(
        sample_dir, epochs, total_batch), skel=(skel+input_size)/(input_size*2))
    print("saved prediction")

def vnect(train_dataset, valid_dataset, resume_train=True):

    if resume_train and os.path.exists('model.json'):
        head_model = load_model()
    else:
        head_model = build_model()

    train_heat_maps, train_img_rgb, test_img_rgb, train_skel, test_skel, n_samples, total_batch = getData(
        train_dataset, valid_dataset, batch_size)

    head_model = train(head_model, train_img_rgb, train_heat_maps)
    predict(head_model, test_img_rgb, total_batch)
    return 1



if __name__ == '__main__':
    if globalConfig.dataset == 'H36M':
        import data.h36m as h36m
        ds = Dataset()
        # ds.loadH36M_expended(64*10, mode='train',
        #                      tApp=True, replace=False)
        ds.loadH36M(64*100, mode='train',
                    tApp=True, replace=False)
        val_ds = Dataset()
        # val_ds.loadH36M_all('all', mode='valid',
                            # tApp=True, replace=False)
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
