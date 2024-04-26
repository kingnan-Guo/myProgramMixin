# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         eager模式Demo
# Author:       Kingnan
# Date:         2020/6/8
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print(tf.executing_eagerly())
x = [[2,]]
m = tf.matmul(x, x)
print(m)
a = tf.constant([[1, 2], [3, 4]])
print(a)
b = tf.add(a, 1)
print(b)
c = tf.multiply(a, b)
print(c)
# 将数字转tensor
num = tf.convert_to_tensor(10)
print('num', num)
for i in range(num.numpy()):
    i = tf.constant(i)
    if int(i%2) == 0:
        print('even')
    else:
        print(' asd')


d = np.array([[5, 6], [7, 8]])

print((a+ b).numpy())