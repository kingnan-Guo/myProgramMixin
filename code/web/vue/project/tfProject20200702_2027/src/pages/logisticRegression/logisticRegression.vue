<template>
  <div class="logisticRegression">
    <h1>逻辑回归</h1>
    <span>logisticRegression</span>
    <div>
      <el-form :inline="true" :model="formInline" class="demo-form-inline">
        <el-form-item label="x">
          <el-input v-model="formInline.x" placeholder="x"></el-input>
        </el-form-item>
        <el-form-item label="y">
          <el-input v-model="formInline.y" placeholder="y"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="onSubmit">查询</el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script>
// import FileSaver from 'file-saver'
import * as tf from '@tensorflow/tfjs'
// import { NewLayer } from '@/assets/tfJs/tensorflow/newLayer'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from '@/assets/resourcePage/js-ml-code/logistic-regression/data.js'
export default {
  name: 'normalization',
  data () {
    return {
      preModel: null,
      formInline: {
        x: '',
        y: ''
      }
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
    },
    addData () {
      const data = getData(400)
      console.log('data ---', data)
      // 生成散点图
      tfvis.render.scatterplot(
        { name: '逻辑回归训练数据' },
        { values: [
          data.filter(item => { return item.label === 1 }),
          data.filter(item => { return item.label === 0 })
        ]}
        // { xAxisDomain: [0, 10], yAxisDomain: [0, 10] }
      )
      const model = tf.sequential()
      model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: 'sigmoid'// 激活函数
      }))
      model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
      })
      const inputs = tf.tensor(data.map(p => [p.x, p.y]))
      const labels = tf.tensor(data.map(p => p.label))
      model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
          { name: '训练效果' },
          ['loss']
        )
      }).then((loss) => {
        this.preModel = model
        // const output = model.predict()
        // model.print()
        // console.log('output == 将tensor转化成数字对象 ', output.mul(20).add(40).dataSync())
      })
    },
    onSubmit () {
      console.log('this.preModel', this.preModel)
      const pred = this.preModel.predict(tf.tensor([[this.formInline.x * 1, this.formInline.y * 1]]))
      console.log('output == 将tensor转化成数字对象 ', pred.dataSync())
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- logisticRegression ------------')
    console.log(tf)
    this.addData()
    // console.log('tfvis --', tfvis)
    // console.log('getData', getData)
  },
  filters: {
  }
}
</script>
