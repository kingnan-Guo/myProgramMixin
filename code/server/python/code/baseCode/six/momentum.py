# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         momentum
# Author:       mac
# Date:         2020/7/10
# Description:  
#
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = 0.01
        self.momentum = momentum
        self.v = None

    # params 是权重
    def update(self, params, grads):
        # v = self.v * self.momentum -
        if self.v == None:
            self.v = {}
            for key, val in params:
                self.v[key] = np.zeros_like(val)

        for key in params.key():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] = params[key] + self.v[key]





params = {
    'w1': [1.0, 0.5, 0.1],
    'b1': [1.0, 2.0, 3.0]
}
grad = {
    'w1': 1.0,
    'b1': 2.0
}