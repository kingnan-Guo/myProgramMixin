# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         数据读取
# Author:       Kingnan
# Date:         2020/6/14
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print (os.path.abspath('.'))

print('TEnsorflow Version: {} '.format(tf.__version__))  #输出版本号


data = pd.read_csv('/Users/kingnan/program/code/server/python/code/tensorflow/tensorflow_base/resourceData/dateset/csv/Income1.csv')
print(data)
