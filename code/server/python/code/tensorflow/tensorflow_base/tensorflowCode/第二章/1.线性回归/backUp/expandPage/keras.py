# _*_　 coding: utf-8 _*_
__author__ = "kingnan"
__date__ = '2020/5/17 18:56'


# import tensorflow as tf


import tensorflow as tf
print(tf.__version__)  #输出版本号
print(tf.test.is_gpu_available())  #安装成功应该输出True

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)



tf.executing_eagerly()
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(A)
print(C)
tf.print('This is my first tensorFlow example: Hello World! Congratulations!!!')



