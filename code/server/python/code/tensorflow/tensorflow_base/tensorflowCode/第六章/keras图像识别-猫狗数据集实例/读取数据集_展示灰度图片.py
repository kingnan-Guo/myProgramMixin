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



# 获取文件夹内的所有文件  ./train
all_train_image_path = data_root.glob('./train/*/*')
# print('all_image_path', len(list(all_image_p ath)))
all_train_image_path_list = list(all_train_image_path)
#将路径变为string 放入到list中
all_train_image_path_str_list = [str(path) for path in all_train_image_path_list]
print(all_train_image_path_str_list[10:20])
# 进行路径乱序 无返回值
random.shuffle(all_train_image_path_str_list)

# sorted 排序
label_name = sorted(item.name for item in data_root.glob('./train/*'))
print('label_name', label_name)
# 将数组转成字典
label_name_to_index_dict = dict((name, index) for name, index in enumerate(label_name))
print('label_name_to_index_dict', label_name_to_index_dict)
#
label_inde_to_name_dict = dict((index, name) for name, index in enumerate(label_name))
print('label_inde_to_name_dict', label_inde_to_name_dict)

# 加载 图片数据
def load_image(image_path):
    # print(image_path.shape)
    image_raw = tf.io.read_file(image_path)
    # print('image_raw', image_raw, '=== image_raw', image_raw.shape)
    # 解码图片
    image_tensor_rgb = tf.image.decode_jpeg(image_raw, channels=1)
    # print('image_tensor_rgb', image_tensor_rgb)
    # 图片转灰度图
    # image_gray = tf.image.rgb_to_grayscale(image_tensor_rgb)
    # 重新定义图片大小
    image_gray_resize = tf.image.resize(image_tensor_rgb, [200, 200])
    print('image_gray_resize --- ', image_gray_resize)

    # image_gray_resize_reshape = (image_gray_resize, [200, 200, 1])
    # print('image_gray_resize_reshape', image_gray_resize_reshape)
    # image_tensor_case = tf.cast(image_gray_resize, tf.uint8)
    # image_tensor_case_normalize = image_tensor_case / 255
    # print('image_tensor_case', np.squeeze(image_tensor_case).shape)
    return np.squeeze(image_gray_resize)

plt.imshow(load_image(all_train_image_path_str_list[10]), cmap='Greys_r')
plt.show()


# 使用 tf 打开图片
train_path_dataset = tf.data.Dataset.from_tensor_slices(all_train_image_path_str_list)
# print('path_dataset ---', path_dataset.take(5))

for image_index in train_path_dataset.take(5):
    print('image_label_index --- ', image_index.numpy())

# 加载图片为tensor

# image_dataset_list = train_path_dataset.map(load_image)


# image_filenames = data_root.glob('./train/*.jpg')
# print('image_filenames', len(list(image_filenames)))