# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         数据读取
# Author:       Kingnan
# Date:         2020/6/14
# Description:  
#
#
#-------------------------------------------------------------------------------

# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setFunction(x):
    y = x > 0
    print(y.astype(np.int))
setFunction(np.array([2, 1, -1]))

def relu(x):
    grad = np.zeros(x)
    print('grad', grad)
    # grad[x>=0] = 1
    # print(grad)
    # return grad

relu(1)


def softmax(x):
    print('x.ndim', x.ndim)
    a = np.exp(x)
    a_sum = np.sum(np.exp(x))
    print('a =', a, 'a_sum =', a_sum, 'a/a_sum =',  a/a_sum)
    return a/a_sum



softmax(np.array([[0.3, 1]]))


def softmax_sencond(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    exp_a_sum = np.sum(np.exp(a))

    y = exp_a/exp_a_sum
    print(y)
    return y

softmax_sencond(np.array([[0.3, 1, -1], [1, 2, 3]]))


def softmax_com(x):
    if x.ndim == 2:
        # x = x.T
        print('x.T ==', x)
        print('np.max(x, axis=0)', np.max(x, axis=0))
        x = x - np.max(x, axis=0)
        print('x==', x)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        print('y ===',y)
        print(y)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

softmax_com(np.array([[0.3, 1], [-1, 33]]))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

print(mean_squared_error(np.array([0, 1, 0]), np.array([0.3, 0.5, 0.2])))


def cross_entropy_error(y, t):
    # 1e-7 = 0.0000001
    # 因为 ln(0) 会变为 无限大的 -inf 所以要加一个微小值
    delta = 1e-7
    print(delta)
    return - np.sum(t * np.log(y + delta))

def creatNpArray(x):
    return np.array(x)


print('cross_entropy_errot(3, 4)', cross_entropy_error(creatNpArray([0, 1, 0]), creatNpArray([0.3, 0.5, 0.2])))


def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        print('y ===', y, 'y.size ==', y.size)
    print('t  == ', t)

    if t.size == y.size:
        t = t.argmax(axis=1)
        # y = y.argmax(axis=1)
    print('t argmax == ', t)

    batch_size = y.shape[0]
    print('batch_size', batch_size, 'np.arange', np.arange(batch_size))

    print('y[np.array(batch_size), t] == ', y[np.arange(batch_size), t])
    print('y[2] == ', y[0, 0])
    # 1e-7 = 0.0000001
    # 因为 ln(0) 会变为 无限大的 -inf 所以要加一个微小值
    delta = 1e-7
    y_input= y[np.arange(batch_size), t]
    log = np.log(y_input + delta)
    sum = np.sum(log)
    return -(sum / batch_size)

print('cross_entropy_errot ', cross_entropy_error_batch(creatNpArray([0.2, 0.1, 0.5, 0.2]), creatNpArray([0, 1, 0, 0])))

def f_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print(grad)
    print('x ---', x.size)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    print('grad ---', grad)
    return grad
numerical_gradient(f_2, np.array([3.0, 4.0]))




def gradient_descent1(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for index in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


print('gradient_descent --', gradient_descent1(f_2, np.array([-3.0, 4.0])))

# draw_log()