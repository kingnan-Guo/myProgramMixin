<template>
  <div class="linear-regression">
    <span>LinearRegression</span>
  </div>
</template>

<script>
// import FileSaver from 'file-saver'
import * as tf from '@tensorflow/tfjs'
// import { NewLayer } from '@/assets/tfJs/tensorflow/newLayer'
// console.log('tf --', tf)
// import * as tfvis from '@tensorflow/tfjs-vis'
import * as tfvis from '@tensorflow/tfjs-vis'
// import {tfvis }  from "@tensorflow/tfjs-vis";
// tfvis.
export default {
  name: 'LinearRegression',
  data () {
    return {
    }
  },
  methods: {
    add () {
      const xs = [1, 2, 3, 4]
      const ys = [1, 3, 5, 7]
      // 创建散点图
      tfvis.render.scatterplot(
        { name: '线性回归' },
        { values: xs.map((x, i) => ({x, y: ys[i]})) },
        { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
      )
      // 创造一个连续的模型，  上一层的输入就是这一层的输出
      const model = tf.sequential()
      // 全连接
      // units 神经元个数
      model.add(tf.layers.dense({units: 1, inputShape: [1]}))
      // 设置损失函数为均方误差 loss: tf.losses.meanSquaredError
      // 优化器 随机梯度下降 SGD  tf.train.sgd(learningRate)  learningRate:学习率
      model.compile({loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.05)})
      const inputs = tf.tensor(xs)
      const labels = tf.tensor(ys)
      const surface = { name: 'show.fitCallbacks', tab: 'Training' }
      model.fit(inputs, labels, {
        batchSize: 4, // 每个次给模型的一个参数
        epochs: 100, // 训练 100 步
        callbacks: tfvis.show.fitCallbacks(
          // {name: '训练过程'},
          // ['loss']
          surface,
          ['loss', 'acc']
        )
      }).then((loss) => {
        const output = model.predict(tf.tensor([5]))
        output.print()
        console.log('output == 将tensor转化成数字对象 ', output.dataSync())
      })
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- LinearRegression ------------')
    console.log(tf)
    this.add()
    console.log('tfvis --', tfvis)
  },
  filters: {
  }
}
</script>
