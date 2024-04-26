# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         twoLayerNet
# Author:       mac
# Date:         2020/7/7
# Description:  
#
#
#-------------------------------------------------------------------------------


import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# OrderedDict 有序字典
from collections import OrderedDict

from softMaxWithLoss import SoftMaxWithLoss
from Layers import Affine, Relu
import sys
# print(sys.path)
# if __name__ == '__main__':
#     print('SoftMaxWithLoss', Affine)
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.lastLayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            # print('layer ---', layer)
            x = layer.forward(x)
            # print('x ---', x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(t == y)/ np.float32(x.shape[0])
        return accuracy


    def gradient(self, x, t):
        # lastLayer的 正向传播
        # forword
        self.loss(x, t)

        #backword
        dout = 1
        dout = self.lastLayer.backward(dout)
        print('lastLayer   dout==', dout.shape)
        layerList = list(self.layers.values())
        layers = reversed(layerList)
        # print('layers --', list(layers))
        for layer in layers:
            print(' ---', layer)
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        print('grads ==', grads)
        return grads


# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
#
# network.predict(1)


