# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         min_net
# Author:       mac
# Date:         2020/7/9
# Description:  
#
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class L_W:
    def __init__(self):
        self.w = [0.5]

    def Y(self, x):
        y = x * self.w[0]
        print('self.w[0] ==', self.w[0])
        return y

    def L(self, x, t):
        y = self.Y(x)
        loss = y-t
        return loss

def G(L, w):
    h = 0.1
    w[0] = w[0] + h
    loss_a =L(w[0])
    w[0] = w[0] - (2 * h)
    loss_b = L(w[0])
    grad = (loss_a - loss_b)/(2 * h)
    return grad

net = L_W()
x_glb = 1
t_glb = 2
f = lambda _:net.L(x_glb, t_glb)
dw = G(f, net.w)
print('dw --', dw)
print('dw --', dw)
