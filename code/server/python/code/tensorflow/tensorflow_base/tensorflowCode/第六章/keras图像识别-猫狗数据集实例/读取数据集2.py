# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         读取数据集
# Author:       Kingnan
# Date:         2020/6/7
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 路径管理
import pathlib
import random

data_dir = 'D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dateset/dog and cat/dc_2000'
data_root = pathlib.Path(data_dir)
print(data_root)


image_filenames = data_root.glob('./train/*.jpg')
print('image_filenames', len(list(image_filenames)))