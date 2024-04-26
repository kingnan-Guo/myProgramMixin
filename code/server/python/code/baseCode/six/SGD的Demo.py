# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         SDG的Demo
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

x_SGD = [-7.0]
y_SGD = [2.0]
x = 7.0
y = 2.0
N = 40
lr = 0.0


for i in range(40):
    x = x - 1/10 * x * 0.9
    y = y - 2 * y * 0.9
    x_SGD.append(x)
    x_SGD.append(y)
