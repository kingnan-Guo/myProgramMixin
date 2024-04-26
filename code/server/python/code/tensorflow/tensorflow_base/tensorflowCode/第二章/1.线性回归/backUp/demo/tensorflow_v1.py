# _*_　 coding: utf-8 _*_
__author__ = "kingnan"
__date__ = '2020/5/17 20:52'


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
a=tf.constant(1)
b=tf.constant(2)
print(sess.run(a+b))
