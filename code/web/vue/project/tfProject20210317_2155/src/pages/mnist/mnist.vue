<template>
  <div class="mnist">
    <h1>mnist</h1>
    <div>加载mnist数据集</div>
    <canvas id="numCanvas" width="300" height="300" style="border: 2px solid #666;"></canvas>
    <div class="btnCox">
      <el-button type="primary" @click="clear()">clear</el-button>
      <el-button type="primary" @click="predict()">predict</el-button>
    </div>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { MnistData } from '@/assets/resourcePage/js-ml-code/mnist/data.js'
// import asd from '@/assets/resourcePage/js-ml-code/data/mnist/mnist_images.png'
export default {
  name: 'mnist',
  data () {
    return {
      preModel: null,
      canvas: null
    }
  },
  methods: {
    async getMnistData () {
      const data = new MnistData()
      console.log('data', data)
      // 加载图片 和 二进制 文件
      await data.load()
      const examples = data.nextTestBatch(20)
      console.log('examples ====', examples)
      examples.labels.print()
      examples.xs.print()

      const x = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3])
      x.print()
      // x.slice([0, 2], [2, 1]).print() [0, 2] 是从第0维 的 键值为2的 -位置开始, 截取 维度为 2 行 1列的数组
      x.slice([1, 1], [0, 2]).print()
      for (var i = 0; i < 20; i++) {
        // tf.tidy() 清除GPU中的 内存
        const imageTensor = tf.tidy(() => {
          // 将数据 reshape 成三维 图片
          return examples.xs.slice([i, 0], [1, (28 * 28)]).reshape([28, 28, 1])
        })
        // imageTensor.print()
        // 在新建 之前 要创建canvas 对象
        const canvas = document.createElement('canvas')
        canvas.width = 28
        canvas.height = 28
        tf.browser.toPixels(imageTensor, canvas).then((val) => {
          // console.log('val ==>', val)
          document.body.appendChild(canvas)
        })
      }
    },
    async drawOnVis () {
      const data = new MnistData()
      console.log('data', data)
      // 加载图片 和 二进制 文件
      await data.load()
      const examples = data.nextTestBatch(20)
      console.log('examples ====', examples)
      examples.labels.print()
      examples.xs.print()
      const surface = tfvis.visor().surface({name: '数字数据输入示例'})
      console.log('surface ==>', surface)
      for (var i = 0; i < 20; i++) {
        // tf.tidy() 清除GPU中的 内存
        const imageTensor = tf.tidy(() => {
          // 将数据 reshape 成三维 图片
          return examples.xs.slice([i, 0], [1, (28 * 28)]).reshape([28, 28, 1])
        })
        // imageTensor.print()
        // 在新建 之前 要创建canvas 对象
        const canvas = document.createElement('canvas')
        canvas.width = 28
        canvas.height = 28
        canvas.style = 'margin: 4px'
        tf.browser.toPixels(imageTensor, canvas).then((val) => {
          // console.log('val ==>', val)
          surface.drawArea.appendChild(canvas)
        })
      }
      // 初始化 网络模型
      const model = tf.sequential()
      // 添加卷积层
      model.add(tf.layers.conv2d({
        // [width, height, channel]
        inputShape: [28, 28, 1],
        // 因为 奇数 的卷积层有中心点， 可以更好的提取特征
        kernelSize: [3, 3],
        filters: 8, // 提取特征的个数
        strides: 1, // 移动步数
        activation: 'relu', // 激活函数
        kernelInitializer: 'varianceScaling'// 卷积核初始化方法， 可以加快收敛速度
      }))
      model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }))
      model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }))
      model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }))
      // 将高维数据转化成一维
      model.add(tf.layers.flatten())
      model.add(tf.layers.dense({
        units: 10, // 十分类
        activation: 'softmax', // 激活函数
        kernelInitializer: 'varianceScaling'
      }))
      model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(),
        metrics: ['accuracy'] // 准确度
      })
      model.summary()
      // 准备训练集
      const [trainXs, trainYs] = tf.tidy(() => {
        const traindata = data.nextTestBatch(1000)
        console.log('d ===>', traindata)
        return [
          traindata.xs.reshape([1000, 28, 28, 1]),
          traindata.labels
        ]
      })
      // 准备验证集
      const [testXs, testYs] = tf.tidy(() => {
        const testdata = data.nextTestBatch(200)
        return [
          testdata.xs.reshape([200, 28, 28, 1]),
          testdata.labels
        ]
      })
      console.log('[trainXs, trainYs]', [trainXs, trainYs])
      model.fit(trainXs, trainYs, {
        epochs: 50,
        batchSize: 500,
        validationData: [testXs, testYs],
        callbacks: tfvis.show.fitCallbacks(
          { name: 'mnist train' },
          ['loss', 'val_loss', 'acc', 'val_acc'],
          { callbacks: ['onEpochEnd'] }
        )
      }).then(() => {
        this.preModel = model
        console.log('predict end')
      })
    },
    bindCanvasFunction () {
      this.canvas.addEventListener('mousemove', (e) => {
        // console.log('addEventListener e', e)
        if (e.buttons === 1) {
          const cxt = this.canvas.getContext('2d')
          cxt.fillStyle = 'rgb(255, 255, 255)'
          cxt.fillRect(e.offsetX, e.offsetY, 15, 15)
        }
      })
    },
    // 清除
    clear () {
      const cxt = this.canvas.getContext('2d')
      cxt.fillStyle = 'rgb(0, 0, 0)'
      cxt.fillRect(0, 0, 300, 300)
    },
    // 预测
    predict () {
      const input = tf.tidy(() => {
        // tf.image.resize_bilinear(images, size, align_corners=False, name=None)
        // 使用双线性插值调整images为size.
        // 输入图像可以是不同的类型,但输出图像总是浮点型的.
        // 参数：
        // images：一个Tensor,必须是下列类型之一：int8,uint8,int16,uint16,int32,int64,bfloat16,half,float32,float64；4维的并且具有形状[batch, height, width, channels].
        // size：2个元素(new_height, new_width)的1维int32张量,用来表示图像的新大小.
        // align_corners：可选的bool,默认为False；如果为True,则输入和输出张量的4个角像素的中心对齐,并且保留角落像素处的值.
        // name：操作的名称(可选).
        // 返回值：
        // 该函数返回float32类型的Tensor.
        const imgageResizeTensor = tf.image.resizeBilinear(
          // 将 canvas 转化成tensor  tf.browser.fromPixels(this.canvas)
          tf.browser.fromPixels(this.canvas),
          [28, 28],
          true
        )
        // 变成 黑白图片 转成 float类型
        const blackAndWhiteInage = imgageResizeTensor.slice([0, 0, 0], [28, 28, 1]).toFloat()
        console.log('blackAndWhiteInage', blackAndWhiteInage)
        // 归一化 div
        const normalizationData = blackAndWhiteInage.div(255)
        console.log('normalizationData', normalizationData.print())
        // 转化 数据格式
        return normalizationData.reshape([1, 28, 28, 1])
      })
      console.log('input ==>', input)
      const pred = this.preModel.predict(input).argMax(1)
      console.log('pred ==>', pred)
      console.log('预测结果为', pred.dataSync()[0])
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- mnist ------------')
    console.log(tf)
    // this.getMnistData()
    // this.egCode()
    console.log('tfvis --', tfvis)
    this.canvas = document.getElementById('numCanvas')
    this.bindCanvasFunction()
    this.drawOnVis()
  },
  filters: {
  }
}
</script>
