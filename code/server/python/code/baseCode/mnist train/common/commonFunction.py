# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         commonFunction
# Author:       mac
# Date:         2020/7/1
# Description:  
#
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt



class commonFunction:
    def __init__(self):
        pass

    def identity_function(x):
        return x

    def step_function(x):
        return np.array(x > 0, dtype=np.int)

    def sigmoid(inX):
        # print('inX --', (inX>0).all())
        # if (inX>0).all():
        #     return 1.0 / (1 + np.exp(-inX))
        # else:
        #     return np.exp(inX) / (1 + np.exp(inX))

        return 1 / (1 + np.exp(-inX))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        grad = np.zeros(x)
        grad[x >= 0] = 1
        return grad

    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))

    def mean_squared_error(y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def softmax_loss(self, X, t):
        y = self.softmax(X)
        return self.cross_entropy_error(y, t)



    def  numerical_gradient(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = np.float32(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 还原值
            it.iternext()

        return grad


    # def numerical_gradient(f, x):
    #     h = 1e-4
    #     grad = np.zeros_like(x)
    #     print(grad)
    #     print('x ---', x.size)
    #     print('np.arange', np.arange(x.size))
    #     print('f ====', f)
    #     for idx in np.arange(x.size):
    #         tmp_val = x[idx]
    #         x[idx] = np.float32(tmp_val) + h
    #         fxh1 = f(x)  # f(x+h)
    #         x[idx] = tmp_val - h
    #         fxh2 = f(x)  # f(x-h)
    #         grad[idx] = (fxh1 - fxh2) / (2 * h)
    #         x[idx] = tmp_val  # 还原值
    #     print('grad ---', grad)
    #     return grad