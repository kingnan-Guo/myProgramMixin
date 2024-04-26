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
BATCH_SIZE = 32

data_dir = 'D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dateset/dog and cat/dc_2000'
data_root = pathlib.Path(data_dir)
print(data_root)



# 获取文件夹内的所有文件  ./train
all_train_image_path = data_root.glob('./train/*/*')
# print('all_image_path', len(list(all_image_p ath)))
all_train_image_path_list = list(all_train_image_path)
#将路径变为string 放入到list中
all_train_image_path_str_list = [str(path) for path in all_train_image_path_list]
# print(all_train_image_path_str_list[10:20])





# 获取 test 数据 -----------------
# 获取文件夹内的所有文件  ./train
all_test_image_path = data_root.glob('./test/*/*')
# print('all_image_path', len(list(all_image_p ath)))
all_test_image_path_list = list(all_test_image_path)
#将路径变为string 放入到list中
all_test_image_path_str_list = [str(path) for path in all_test_image_path_list]
# ---------------------------------




# 进行路径乱序 无返回值
# random.shuffle(all_train_image_path_str_list)

# sorted 排序
label_name = sorted(item.name for item in data_root.glob('./train/*'))
print('label_name', label_name)
# 将数组转成字典
label_index_to_name_dict = dict((name, index) for name, index in enumerate(label_name))
print('label_index_to_name_dict', label_index_to_name_dict)
#
label_name_to_index_dict = dict((index, name) for name, index in enumerate(label_name))
print('label_name_to_index_dict', label_name_to_index_dict)

# 加载 图片数据
def load_image(image_path, label_index):
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

    image_gray_resize_reshape = tf.reshape(image_gray_resize, [200, 200, 1])
    # print('image_gray_resize_reshape', image_gray_resize_reshape)
    image_tensor_normalize = image_gray_resize_reshape / 255
    print('image_tensor_case_normalize', image_tensor_normalize)

    # print('image_path ========', list(image_path))
    # label_index =
    return image_tensor_normalize, label_index

# plt.imshow(np.squeeze(load_image(all_train_image_path_str_list[10])), cmap='Greys_r')
# plt.show()

train_label_index_list = list(label_name_to_index_dict[pathlib.Path(image_path).parent.name] for image_path in all_train_image_path_str_list)

# 使用 tf 打开 train图片
train_path_dataset = tf.data.Dataset.from_tensor_slices((all_train_image_path_str_list, train_label_index_list))
# 加载图片为tensor
image_tensor_normalize = train_path_dataset.map(load_image)
# print('image_tensor_case_normalize --- ', image_tensor_case_normalize)

# 加载 test 图片 和label
test_label_index_list = list(label_name_to_index_dict[pathlib.Path(image_path).parent.name] for image_path in all_test_image_path_str_list)
test_path_dataset = tf.data.Dataset.from_tensor_slices((all_test_image_path_str_list, test_label_index_list))


image_tensor_normalize_shuffle = image_tensor_normalize.shuffle(buffer_size=2000)
image_tensor_normalize_shuffle_repeat = image_tensor_normalize_shuffle.repeat()
image_tensor_normalize_shuffle_repeat_batch = image_tensor_normalize_shuffle_repeat.batch(BATCH_SIZE)
print('image_tensor_normalize_shuffle_repeat_batch', image_tensor_normalize_shuffle_repeat_batch)

test_image_tensor_batch = test_path_dataset.batch(BATCH_SIZE)

train_count = len(all_train_image_path_str_list)
test_count = len(all_test_image_path_str_list)
steps_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE

# history = model.fit(image_tensor_normalize_shuffle_repeat_batch, epochs=30, steps_per_epoch=steps_per_epoch, validation_data=test_dataset_batch, validation_steps=validation_steps)
#
#
# plt.plot(history.epoch, history.history.get('acc'), label='acc')
# plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
# plt.legend()
# plt.show()