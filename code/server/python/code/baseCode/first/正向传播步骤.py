# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         数据读取
# Author:       Kingnan
# Date:         2020/6/14
# Description:  
#   当前遇到的问题是 W 改变了， 对整体有什么影响，为什么说W改变后f（x）就可以改变，就可以求出 grad
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    if x.ndim == 2:
        # x = x.T
        print('x.T ==', x)
        print('np.max(x, axis=0)', np.max(x, axis=0))
        x = x - np.max(x, axis=0)
        print('x==', x)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        print('y ===',y)
        print(y)
        return y.T
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    # 1e-7 = 0.0000001
    # 因为 ln(0) 会变为 无限大的 -inf 所以要加一个微小值
    delta = 1e-7
    return - np.sum(t * np.log(y + delta))


def f_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print(grad)
    print('x ---', x.size)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    print('grad ---', grad)
    return grad

class simpleNet:
    def __init__(self):
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.w)

    def loss(self, x, t):
        print(' loss w ==', self.w)
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss



def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            print('enumerate ---', x)
            grad[idx] = _numerical_gradient_1d(f, x)
        return grad

net = simpleNet()
print(net.w)
x = np.array([0.6, 0.9])
p = net.predict(x)
print('p ---', p)
t = np.array([0, 0, 1])
loss = net.loss(x, t)
print('loss --', loss)

f = lambda w: net.loss(x, t)
dw = numerical_gradient_2d(f, net.w)
print('dw ----', dw)

print('- - - ' * 10)

new_net = simpleNet()
print('new_net.w =', new_net.w)
def f_w(w):
    print('f_w   w ---', w, 'net w', new_net.w)
    # new_net.w = w
    return new_net.loss(x, t)
dw_w = numerical_gradient_2d(f_w, new_net.w + 1)

print('dw_w ----', dw_w)
