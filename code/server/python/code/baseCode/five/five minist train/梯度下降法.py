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
import numpy as np
import matplotlib.pyplot as plt
from twoLayerNet import TwoLayerNet
np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':
    print('twoLayerNet', TwoLayerNet)
    # from dataset.mnist import load_mnist
    # tf.keras.datasets.mnist()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    y_train = np.eye(10)[y_train]
    # print(y_train, '--------')
    y_test = np.eye(10)[y_test]

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num= 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    for i in range(10):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = y_train[batch_mask]


        print('x_batch =', x_batch.shape, 't_batch =', t_batch.shape)
        # 计算梯度
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)


        # print('grad ----', grad)
        for key in ('w1', 'b1', 'w2', 'b2'):
            print( 'key', key,' ---grad[key]',grad[key].shape)
            print('network.params[key]', network.params[key].shape)
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch:
            train_acc = network.accuracy(x_train, y_train)
            test_acc = network.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('train_acc = ' + str(train_acc) + "test_acc = " + str(test_acc))






    # /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/keras/datasets


    # 绘制图形
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()