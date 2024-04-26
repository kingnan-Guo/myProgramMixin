# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         卷积综合实例_数据预处理2
# Author:       Kingnan
# Date:         2020/6/1
# Description:  
#
#
#-------------------------------------------------------------------------------


import tensorflow as tf
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 路径管理
import pathlib
import random

data_dir = 'D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dateset/photo/2_class'
data_root = pathlib.Path(data_dir)
print(data_root)
# iterdir 对目录进行迭代, 获取文件夹下所有的 文件夹
data_root_iterdir = data_root.iterdir()
print(list(data_root_iterdir))
# for item in data_root_iterdir:
#     print('文件夹 --', item)
# print('data_root.iterdir()', data_root_iterdir)


# glob 要输入正则表达式， */* 所有文件夹下的所有 文件
all_image_path = data_root.glob('*/*')
all_image_path_list = list(all_image_path)
print('all_image_path len==', len(all_image_path_list))
print(all_image_path_list[:3])
# 将路径变成 string
all_image_path_str = [str(path) for path in  all_image_path_list]
print('all_image_path_str', all_image_path_str[10:12])

# 进行路径乱序 无返回值
random.shuffle(all_image_path_str)
print(all_image_path_str[10:12])

#  sorted : Return a new list containing all items from the iterable in ascending order.
label_names = sorted(item.name for item in data_root.glob('*/'))
print('label_names', label_names)

# 将数组 转成字典
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print('label_to_index', label_to_index)

# pathlib.Path(path).parent.name ; 这是每个 图像文件 父级文件夹的 名字 = airplane || lake
# label_to_index[pathlib.Path(path).parent.name] = 1 || 0
all_image_label_index_list = [label_to_index[pathlib.Path(path).parent.name] for path in  all_image_path_str]
print('all_image_label_list', all_image_label_index_list[:5])

index_to_label_dict = dict((value, key) for (key, value) in label_to_index.items())
print('index_to_label_dict', index_to_label_dict)


def load_preprosess_image(image_path):
    print('load_preprosess_image image_path', image_path.shape)
    # tf 读取图片 tf.io.read_file 显示编码形式
    image_raw = tf.io.read_file(image_path)
    # 解码图片
    image_tensor = tf.image.decode_image(image_raw)
    print('image_tensor', image_tensor.shape, '-- image_tensor.dtype ==', image_tensor.dtype)
    # 转成 float32 格式
    image_tensor_case = tf.cast(image_tensor, tf.float32)
    # 标准化 /255
    image_tensor_case_normalize = image_tensor_case / 255
    # print('image_tensor_case', image_tensor_case)
    # print('image_tensor_case_normalize', image_tensor_case_normalize.numpy())
    return image_tensor_case_normalize

# plt.imshow(load_preprosess_image(all_image_path_str[10]))
# plt.show()

# path_dataset
path_dataset = tf.data.Dataset.from_tensor_slices(all_image_path_str)
print('path_ds --- ', path_dataset.take(5))
image_dataset = path_dataset.map(load_preprosess_image)

for image_index in path_dataset.take(5):
    print('image_label_index --- ', image_index.numpy())

label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label_index_list)
print('label_ds --- ', label_dataset)
for image_label_index in label_dataset.take(5):
    print('image_label_index', image_label_index.numpy())

