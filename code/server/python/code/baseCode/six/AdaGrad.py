# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         AdaGrad
# Author:       mac
# Date:         2020/7/10
# Description:  学习率衰减
#  h = dL_dw ** 2; h中保存了所有的 梯度的 平方和
#  1 / √h 当 h 越大时
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdaGrad:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, value in params:
                self.h[key] = np.zeros_like(value)

        for key in params.keys():
            self.h[key] = self.h[key] + grads[key]**2
            params[key] = params[key] - self.lr *  grads[key] * (1 / np.sqrt(self.h + 1e-7)  )


