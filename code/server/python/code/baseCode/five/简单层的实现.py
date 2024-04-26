# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         简单层的实现
# Author:       mac
# Date:         2020/7/4
# Description:  
#
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MulLlayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out;

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dy, dx;






