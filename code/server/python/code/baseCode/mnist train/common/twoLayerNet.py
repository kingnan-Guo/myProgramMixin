# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         twoLayerNet
# Author:       mac
# Date:         2020/6/27
# Description:  双层神经网络
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from commonFunction import commonFunction


class twoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = commonFunction.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = commonFunction.softmax(a2)

        return y


    def loss(self, x, t):
        y = self.predict(x)
        return commonFunction.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        print('y = ', y.shape )
        print('t = ', t.shape)
        y = np.argmax(y, axis=1)
        print('y = ', y.shape )
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / np.float32(x.shape[0])
        return accuracy

    def  numerical_gradient(self, x, t):
        # loss_W = lambda w: self.loss(x, t)
        loss_W = lambda _: self.loss(x, t)
        grad = {}

        grad['w1'] = commonFunction.numerical_gradient(loss_W, self.params['w1'])
        grad['b1'] = commonFunction.numerical_gradient(loss_W, self.params['b1'])
        grad['w2'] = commonFunction.numerical_gradient(loss_W, self.params['w2'])
        grad['b2'] = commonFunction.numerical_gradient(loss_W, self.params['b2'])
        # print('grad --', grad)
        return grad

    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        grads = {}
        batch_num = x.shape[0]

        #
        a1 = np.dot(x, w1) + b1
        z1 = commonFunction.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = commonFunction.softmax(a2)

        # 从这里之后就不知道是要 做什么了
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(z1.T, dy)
        dz1 = commonFunction.sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

