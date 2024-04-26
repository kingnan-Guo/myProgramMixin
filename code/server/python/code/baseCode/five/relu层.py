# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         relu层
# Author:       mac
# Date:         2020/7/5
# Description:  
#
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print('x ==', x)
mask = (x <= 0)
print('mask', mask)
x_copy = x.copy()
print('x_copy[mask]', x_copy[mask])
x_copy[mask] = 0
print('x_copy[mask] =0 ', x_copy[mask])
print(x_copy)