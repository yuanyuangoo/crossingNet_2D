import numpy as np
import sys
import os
import time
sys.path.append('./')
from data.util import *
import globalConfig


def load_data():
    test_skel = []
    test_label = []
    result_1 = []
    result_2 = []
    for i in range(8):
        skel = np.load('test_skel_{}.out.npy'.format(i))
        label = np.load('test_label_{}.out.npy'.format(i))
        result_1_ = np.load('result_{}_{}.out.npy'.format(49, i))
        result_2_ = np.load('result_{}_{}.out.npy'.format(499, i))

        test_skel.append(np.asarray(skel))
        test_label.append(np.asarray(label))
        result_1.append(np.asarray(result_1_))
        result_2.append(np.asarray(result_2_))

    return np.asarray(test_skel), np.asarray(test_label), np.asarray(result_1), np.asarray(result_2)

test_skel, test_label, result_1, result_2 = load_data()
pck = eval_pck((result_1+128)/256, test_skel, test_label)
np.save('result_pck_{}.out'.format(49), pck)
print(pck[:, :, :-1, :-1])
pck = eval_pck((result_2+128)/256, test_skel, test_label)
np.save('result_pck_{}.out'.format(499), pck)
print(pck[:, :, :-1, :-1])
