<template>
  <div class="xor">
    <h1>非线性分类-xor</h1>
    <span>xor</span>
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
import { getData } from '@/assets/resourcePage/js-ml-code/xor/data.js'
export default {
  name: 'Xor',
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
    addData () {
      const data = getData(400)
      console.log('data', data)
      tfvis.render.scatterplot(
        {name: '非线性分类-xor'},
        {
          values: [
            data.filter(p => p.label === 1),
            data.filter(p => p.label === 0)
          ]
        }
      )
      const model = tf.sequential()
      model.add(tf.layers.dense({
        units: 4,
        inputShape: [2],
        activation: 'relu'
      }))
      // 内部的 inputShape 会自动计算不用设置
      // 因为是二分尅所以只能用 sigmoid
      // 输出层
      model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
      }))
      model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
      })
      const inputs = tf.tensor(data.map(p => [p.x, p.y]))
      const labels = tf.tensor(data.map(p => p.label))
      model.fit(inputs, labels, {
        // batchSize: 40, // 默认 32
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
          { name: '非线性分类-xor' },
          ['loss']
        )
      }).then((loss) => {
        this.preModel = model
        console.log('训练完毕')
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
    console.log('------------- xor ------------')
    console.log(tf)
    this.addData()
  },
  filters: {
  }
}
</script>
