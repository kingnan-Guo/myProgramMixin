# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         
# Author:       mac
# Date:         2020/7/8
# Description:  
#
#
# -------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


a = np.arange(6).reshape(2, 3)
print('a ==', a)
for x in np.nditer(a):
    print('x ==', x)

# order='F' 按照列访问 控制迭代顺序
a = np.arange(6).reshape(2, 3)
print('a ==\n', a)
for x in np.nditer(a, order='F'):
    print('x ==', x)

# 转秩 后 不会影响迭代的 顺序
# 修改 order='C' 按照行访问    会修改迭代顺序
b = a.T
print('b ===\n', b)
for x in np.nditer(b):
    print('x == ', x)
for x in np.nditer(b, order='C'):
    print('x F== ', x)



# flags = 【
# 对 a 进行多行索引
# 元素的坐标
# it.iternext() 进入下一次迭代
# op_flags=['readwrite'] 可读写
it = np.nditer(a, flags=['multi_index'], op_flags=['readwrite'])
print('it ==', it)
while not it.finished:
    print(it.multi_index)
    it.iternext()