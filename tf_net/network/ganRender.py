import scipy.optimize
from forwardRender import ForwardRender
import cv2
import time
import shutil
import os
import json
import numpy.linalg
from numpy.random import RandomState

from collections import namedtuple
import sys
sys.path.append('./')
import globalConfig
from data.dataset import *

class GanRender(ForwardRender):
    DisErr = namedtuple('disErr', ['gan', 'est', 'metric'])
    GenErr = namedtuple('genErr', ['gan', 'recons', 'metric'])
    golden_max = 1.0 

    def __init__(self, x_dim, rndGanInput=False, metricCombi=False):
        super(GanRender, self).__init__(x_dim)
        self.rndGanInput = rndGanInput
        self.metricCombi = metricCombi
    def genLoss(self,isTrain=True):
        real_render_var=self.render
        fake_render_var
        recons_loss = (real_render_var - fake_render_var)**2
        recons_loss = T.clip(recons_loss, 0, self.golden_max)