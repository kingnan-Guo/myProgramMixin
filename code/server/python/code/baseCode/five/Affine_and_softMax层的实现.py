# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         Affine/softMax层的实现
# Author:       mac程序不能买
# Date:         2020/7/5
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.X = x
        out = np.dot(self.X, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        # db 的导数 为什么是 dy 在第0维度的导数
        db = np.sum(dout, axis=0)
        return dx





# 正常B 的维度应该为 （2, 3） ， 但是numpy自动对其进行了扩展
x_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
print('x_dot_W + B = \n', x_dot_W + B)

dy = np.array([[1, 2, 3], [4, 5, 6]])
dB_0 = np.sum(dy, axis=0)
dB_1 = np.sum(dy, axis=1)
print('dB =', dB_0)
