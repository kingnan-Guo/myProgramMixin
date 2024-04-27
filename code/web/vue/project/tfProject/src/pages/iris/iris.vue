<template>
  <div class="xor">
    <h1>鸢尾花-iris</h1>
    <span>iris</span>
    <div>
      <el-form :inline="true" :model="formInline" class="demo-form-inline">
        <el-form-item label="花萼长度：">
          <el-input v-model="formInline.a" placeholder="花萼长度："></el-input>
        </el-form-item>
        <el-form-item label="花萼宽度：">
          <el-input v-model="formInline.b" placeholder="花萼宽度："></el-input>
        </el-form-item>
        <el-form-item label="花瓣长度：">
          <el-input v-model="formInline.c" placeholder="花瓣长度："></el-input>
        </el-form-item>
        <el-form-item label="花瓣宽度：">
          <el-input v-model="formInline.d" placeholder="花瓣宽度："></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="onSubmit">预测</el-button>
        </el-form-item>
      </el-form>
    </div>

  </div>
</template>

<script>

import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getIrisData, IRIS_CLASSES } from '@/assets/resourcePage/js-ml-code/iris/data.js'
// import { model } from '@tensorflow/tfjs'
// import { model, Optimizer } from '@tensorflow/tfjs'
export default {
  name: 'Xor',
  data () {
    return {
      preModel: null,
      formInline: {
        a: '',
        b: '',
        c: '',
        d: ''
      }
    }
  },
  methods: {
    addData () {
      const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15)
      console.log('[xTrain, yTrain, xTest, yTest]', [xTrain, yTrain, xTest, yTest])
      // xTrain.print()
      // yTrain.print()
      // xTest.print()
      // yTest.print()
      const model = tf.sequential()
      model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: 'sigmoid'
      }))
      model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
      }))
      // categoricalCrossentropy 交叉熵损失函数
      // metrics：度量     accuracy：准确度
      model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']
      })
      model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest], // 验证集
        callbacks: tfvis.show.fitCallbacks(
          { name: '鸢尾花训练效果' },
          // val_loss验证集的 loss， acc 准确度， val_acc验证集准确度
          ['loss', 'val_loss', 'acc', 'val_acc'],
          // 只展示 onEpochEnd 的图
          { callbacks: ['onEpochEnd'] }
        )
      }).then(() => {
        console.log('trian end')
        this.preModel = model
      })
    },
    onSubmit () {
      const input = tf.tensor([[
        this.formInline.a * 1,
        this.formInline.b * 1,
        this.formInline.c * 1,
        this.formInline.d * 1
      ]])
      const pred = this.preModel.predict(input)
      // argMax 输某个维度的最大值
      console.log('output == 鸢尾花的类型是 ', IRIS_CLASSES[pred.argMax(1).dataSync(0)])
    }

  },
  watch: {
  },
  mounted () {
    console.log('------------- iris ------------')
    console.log('tf', tf)
    console.log('tfvis', tfvis)
    console.log('IRIS_CLASSES', IRIS_CLASSES)
    this.addData()
  },
  filters: {
  }
}
</script>
<style>
  .demo-form-inline{
    width: 20vw;
  }
</style>
