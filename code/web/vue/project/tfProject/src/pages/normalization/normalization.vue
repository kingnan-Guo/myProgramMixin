<template>
  <div class="normalization">
    <span>归一化normalization</span>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
// import { NewLayer } from '@/assets/tfJs/tensorflow/newLayer'
import * as tfvis from '@tensorflow/tfjs-vis'

export default {
  name: 'normalization',
  data () {
    return {
    }
  },
  methods: {
    add () {
      const height = [150, 160, 170]
      const weight = [40, 50, 60]
      // 创建散点图
      tfvis.render.scatterplot(
        { name: '身高体重预测' },
        { values: height.map((x, i) => ({x, y: weight[i]})) },
        { xAxisDomain: [140, 180], yAxisDomain: [30, 70] }
      )
      // subStrict 减法 div 除法
      // 数据归一化 所有数据减去最小值 然后减去再大值最小值的差
      // 简单缩放 x ＝ (x - min)/(max - min)
      const inputs = tf.tensor(height).sub(150).div(20)
      const labels = tf.tensor(weight).sub(40).div(20)
      inputs.print()
      labels.print()
      console.log('inputs ===>', inputs.dataSync())
      console.log('inputs ===>', labels.dataSync())
      // 创造一个连续的模型，  上一层的输入就是这一层的输出
      const model = tf.sequential()
      // 全连接
      // units 神经元个数
      model.add(tf.layers.dense({units: 1, inputShape: [1]}))
      // 设置损失函数为均方误差 loss: tf.losses.meanSquaredError
      // 优化器 随机梯度下降 SGD  tf.train.sgd(learningRate)  learningRate:学习率
      model.compile({loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.05)})
      model.summary()
      model.fit(inputs, labels, {
        batchSize: 3, // 每个次给模型的一个参数
        epochs: 100, // 训练 100 步
        callbacks: tfvis.show.fitCallbacks(
          {name: '训练过程'},
          ['loss']
        )
      }).then((loss) => {
        const output = model.predict(tf.tensor([180]).sub(150).div(20))
        output.print()
        console.log('output == 将tensor转化成数字对象 ', output.mul(20).add(40).dataSync())
      })
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- normalization ------------')
    console.log(tf)
    this.add()
    console.log('tfvis --', tfvis)
  },
  filters: {
  }
}
</script>
