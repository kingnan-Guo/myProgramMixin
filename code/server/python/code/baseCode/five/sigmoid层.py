# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         sigmoid层
# Author:       mac
# Date:         2020/7/5
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # 反向传播的 第一层 dout * dx ，
        dx = dout * self.out * ( 1 - self.out)
        return dx

