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
    #  image_tensor = tf.image.decode_image(image_raw)  decode_image 这种是通用的解码方式，但是不会返回数据的 shape

    # decode_jpeg 解析指定格式的 图片数据；这样才可以 resize 数据 ； decode_image 后resize 会报错
    image_tensor = tf.image.decode_jpeg(image_raw, channels=3)
    print('image_tensor', image_tensor.shape, '-- image_tensor.dtype ==', image_tensor.dtype)
    # tf.image.resize_image_with_crop_or_pad() 把图片 resize 成指定的 尺寸 但是不会更改图片的 比例
    # 因为此处的图片都是 256 *256 所以 可以直接使用 resize() 只控制 高和宽
    image_tensor_resize = tf.image.resize(image_tensor, [256, 256])
    print('image_tensor_resize --- ', image_tensor_resize)
    # 转成 float32 格式
    image_tensor_case = tf.cast(image_tensor_resize, tf.float32)
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


#
# zip 将两个数据进行打包

dataset_zip = tf.data.Dataset.zip((image_dataset, label_dataset))
print('dataset_zip', dataset_zip)

image_count = len(all_image_path_list)
# 划分测试数据 和 训练数据
test_count = int(image_count*0.2)
train_count = image_count - test_count
# .skip(num) 跳过一定数量的 数据 加载
train_dataset = dataset_zip.skip(test_count)
# take 是取一定的数值
test_dataset = dataset_zip.take(test_count)


# 期望的数据 为乱序 不断重复 并且是一个 batch 接着 一个 batch
BATCH_SIZE = 32
# repeat 源源不断的产生数据    shuffle(buffer_size) 乱序 buffer_size:在多大的范围内 乱序         batch
train_dataset_repeat_shuffle_batch = train_dataset.repeat().shuffle(buffer_size=train_count).batch(BATCH_SIZE)

test_dataset_batch = test_dataset.batch(BATCH_SIZE)


# train_data = train_data.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=train_count))
# train_data = train_data.batch(BATCH_SIZE)
# train_data = train_data.prefetch(buffer_size=AUTOTUNE)




model = tf.keras.Sequential()   #顺序模型
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3)))
# 批标准化 放在卷积层后面
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())


model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'))
# model.add(tf.keras.layers.GlobalAveragePooling2D()) 数据转换成2维的形式
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)

steps_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE

history = model.fit(train_dataset_repeat_shuffle_batch, epochs=30, steps_per_epoch=steps_per_epoch, validation_data=test_dataset_batch, validation_steps=validation_steps)


plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()