

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python.keras import Sequential
csvPath = 'D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dateset/csv'
data = pd.read_csv(csvPath + '/Advertising.csv')
print(data.head())

# TV,radio,newspaper,sales
# %matplotlib inline
plt.scatter( x=data.TV, y=data.sales)
# plt.show()

x = data.iloc[:, 1:-1]
# 最后一列是 销售额
y = data.iloc[:, -1]

print(x,y)


model: Sequential = tf.keras.Sequential(

    [
        # 第一层 输入层
        tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
        # 第二层输出层
        tf.keras.layers.Dense(1)
    ],
)
model.summary()

# Param  = 40 ,因为 每个神经元 有三个输入， (x1 * w1 + x2 * w2 + x3 * w3) + b , (x1, x2, x3, b) 四个参数
# 输出参数 有 11个 ， 因为有10个 神经元  ， 10个输出 + 1个偏置 一共11个

# 训练模型
# 配置优化器  adam ， loss 损失函数 ，均方差 mse
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(x, y, epochs=100)
pred = model.predict(data.iloc[:10, 1:-1])
print('pred ===', pred)


