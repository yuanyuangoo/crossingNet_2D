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
        if not os.path.exists('result_{}_{}.npy'.format(189, i)):
            print('result_{}_{}.out.npy donot exist'.format(189, i))
            continue
        skel = np.load('test_skel_{}.out.npy'.format(i))[0]
        label = np.load('test_label_{}.out.npy'.format(i))[0]
        result_1_ = np.load('result_{}_{}.npy'.format(9, i))
        result_2_ = np.load('result_{}_{}.npy'.format(189, i))

        result_1_=result_1_.reshape((-1,result_1_.shape[2]))
        result_2_=result_2_.reshape((-1,result_2_.shape[2]))

        for idx in range(len(skel)):
            test_skel.append(np.asarray(skel[idx]))
            test_label.append(np.asarray(label[idx]))
            result_1.append(np.asarray(result_1_[idx]))
            result_2.append(np.asarray(result_2_[idx]))

    return np.asarray(test_skel), np.asarray(test_label), np.asarray(result_1), np.asarray(result_2)
pck=[]
test_skel, test_label, result_1, result_2 = load_data()

pck1_3d,pck1_2d = eval_pck(result_1, test_skel*256-128, test_label)

pck2_3d,pck2_2d = eval_pck((result_2+128)/256, test_skel, test_label)

pck.append(pck1_3d)
pck.append(pck1_2d)
pck.append(pck2_3d)
pck.append(pck2_2d)
pck=pck
np.save('result_pck',pck)


pck=np.load('result_pck.npy')
from pprint import pprint as pp
pp(pck[:,:,-1,-1])