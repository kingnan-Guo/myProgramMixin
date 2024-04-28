<template>
  <div class="brand">
    <h1>Transfer Learning </h1>
    <span>brand</span>
    <!-- <img :src='imgSrc'  width='224' height='224' /> -->
    <!-- <img src='./static/data/brand/train/android-0.jpg'  width='224' height='224' /> -->
    <div class="imageBox" v-html="imgSrc"></div>
    <div class="box">
      <el-button size="small" type="primary" @click="getInputs()" >getInputs</el-button>
      <el-upload
        class="upload-demo"
        action="https://jsonplaceholder.typicode.com/posts/"
        :on-preview="handlePreview"
        :on-remove="handleRemove"
        :before-remove="beforeRemove"
        multiple
        :limit="3"
        :on-exceed="handleExceed"
        :before-upload="beforeUpload"
        :file-list="fileList">
        <el-button size="small" type="primary">点击上传</el-button>
        <div slot="tip" class="el-upload__tip">只能上传jpg/png文件，且不超过500kb</div>
      </el-upload>
    </div>
  </div>
</template>

<script>

import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
// import { layer } from '@tensorflow/tfjs-vis/dist/show/model'
// import train_box from '@/assets/resourcePage/js-ml-code/data/brand/train'
export default {
  name: 'Mobilenet',
  data () {
    return {
      preModel: null,
      MOBILENET_MODEL_PATH: '../static/data/mobilenet/web_model/model.json',
      fileList: [],
      imgSrc: null,
      model: null,
      imageSize: 224,
      imageUrlList: [],
      MobileNet: null,
      NUM_CLASS: 3,
      truncatedMobilenet: null
    }
  },
  methods: {
    getInputs () {
      var loadimages = []
      var labelArr = []
      var labelDir = {}
      const requireContext = require.context('@/assets/resourcePage/js-ml-code/data/brand/train', true, /\.jpg$/)
      // 批量读取路径下的文件
      requireContext.keys().map(key => {
        // 获取所有图片额信息
        const image = requireContext(key)
        // 将所有的 promis 存入 数组
        loadimages.push(this.readImgList(image))
        // console.log('readImgList key', key)
        var label = key.split('-')[0]
        // 假设不知道有多少的 类型
        if (!labelDir[label]) {
          labelDir[label] = 0
        }
        labelDir[label] += 1
        console.log('labelDir ', labelDir, 'labelDir.length', Object.getOwnPropertyNames(labelDir))
        var keyArr = Object.getOwnPropertyNames(labelDir)
        var zeroArr = this.formatArr([], this.NUM_CLASS)
        var labelIndex = keyArr.indexOf(label)
        // 在 数组的指定 index 中添加 1
        zeroArr[labelIndex] = 1
        labelArr.push(zeroArr)
      })
      // console.log('loadimages', loadimages)
      // console.log('labelArr ====', labelArr)
      // 处理 promise
      Promise.all(loadimages).then((result) => {
        // 返回所有的图片对象
        console.log('result ====', result)
        this.drawOnVis(result)
        this.changeLayers(result, labelArr)
      })
    },
    // 创建数组
    formatArr (arr, num) {
      let len = arr.length
      if (num <= arr.length) {
        arr.slice(0, num)
      }
      return arr.concat(new Array(num - len).fill(0))
    },
    readImgList (src) {
      return new Promise(resolve => {
        const image = new Image()
        image.src = src
        image.width = this.imageSize
        image.height = this.imageSize
        image.onload = () => resolve(image)
      })
    },
    // 绘制一下 所有的 图片
    drawOnVis (inputs, labels) {
      const surface = tfvis.visor().surface({name: 'show all logo', styles: { height: 100 }})
      inputs.forEach(image => {
        // console.log('labelArr ====', image)
        surface.drawArea.appendChild(image)
      })
    },
    // 加载 mobileNet
    async addModel () {
      await tf.loadLayersModel(this.MOBILENET_MODEL_PATH).then((model) => {
        this.mobileNet = model
        model.summary()
        // this.changeLayers(model)
        return model
      })
    },
    // 截断 mobileNet
    async changeLayers (inputsArr, labelArr) {
      await this.addModel()
      // getLayer 输入层的名称
      var convPw13ReluLayer = this.mobileNet.getLayer('conv_pw_13_relu')
      console.log('conv_pw_13_relu_layer', convPw13ReluLayer)
      // 截断 层
      // 1、创建模型
      // 输入就是 mobileNet 的输入
      // 输出为 截断层
      const truncateMobileNet = tf.model({
        inputs: this.mobileNet.inputs,
        outputs: convPw13ReluLayer.output
      })
      this.truncatedMobilenet = truncateMobileNet
      console.log('conv_pw_13_relu_layer ========', convPw13ReluLayer.outputShape)
      console.log('truncateMobileNet', truncateMobileNet)
      // console.log('convPw13ReluLayer.outputShape.slice(1) ========', convPw13ReluLayer.outputShape.slice(1))
      var model = tf.sequential()
      // 替换全连接层
      model.add(tf.layers.flatten({
        inputShape: convPw13ReluLayer.outputShape.slice(1)
      }))
      model.add(tf.layers.dense({units: 10, activation: 'relu'}))
      model.add(tf.layers.dense({units: this.NUM_CLASS, activation: 'softmax'}))
      model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam()
      })
      // 先把数据 输入到截断模型中
      // 把数据转化成 截断模型需要的 型式
      const {xs, ys} = tf.tidy(() => {
        const imagePredictTensor = inputsArr.map((image) => {
          // 图片处理
          const imageTensor = this.imageTox(image)
          console.log('imageTensor ==', imageTensor)
          // 将数据 传入到 截断模型中
          return truncateMobileNet.predict(imageTensor)
        })
        // 将装满数组的 tensor 合并成一个 tensor ？？？
        console.log('imagePredictTensor', imagePredictTensor)
        // imagePredictTensor.print()
        const xs = tf.concat(imagePredictTensor)
        console.log('xs ===', xs)
        const ys = tf.tensor(labelArr)
        console.log('ys ===', ys)
        ys.print()
        return {xs, ys}
      })
      console.log('return  {xs, ys}', {xs, ys})
      await model.fit(xs, ys, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
          {name: 'train '},
          ['loss'],
          {callbacks: ['onEpochEnd']}
        )
      })
      this.model = model
      console.log('-- train end --', this.model)
    },
    // 图片数据处理
    imageTox (imgData) {
      console.log('imageTox  --------- imgData ', imgData)
      const pred = tf.tidy(() => {
        // console.log('imageTox  --------- tf.browser.fromPixels(imgData).toFloat().div(255 / 2) ',tf.browser.fromPixels(imgData).toFloat().div(255 / 2).reshape([1, 224, 224, 3]))
        // 为了归一化 -255 但是数据 要求 （-1， 1） 所以，先减去 （255/2）再除以（）
        const input = tf.browser.fromPixels(imgData).toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3])
        console.log('imageTox  --------- ', input)
        return input
      })
      return pred
    },
    // 上传图片
    beforeUpload (file) {
      console.log('file', file)
      // return false
      // return isJPG && isLt2M;
      this.readImg(file, (imgData) => {
        const pred = tf.tidy(() => {
          // 为了归一化 -255 但是数据 要求 （-1， 1） 所以，先减去 （255/2）再除以（）
          const input = this.imageTox(imgData)
          console.log(input)
          // var predreshape = input.reshape([1, 224, 224, 3])
          // predreshape.print()
          // console.log(predreshape)
          // return this.model.predict(input)
          const truncatedMobilenetOutPut = this.truncatedMobilenet.predict(input)
          return this.model.predict(truncatedMobilenetOutPut)
        })
        pred.print()
        const index = pred.argMax(1).dataSync()[0]
        console.log('index ==', index)
        console.log('pred ==', pred)
        console.log('pred.argMax(1)', pred.argMax(1))
        // console.log(IMAGENET_CLASSES[index])
      })
    },
    readImg (file, callback) {
      var reader = new FileReader()
      console.log('reader', reader)
      reader.readAsDataURL(file)
      reader.onload = (e) => {
        // console.log('e.target.result', e.target.result);
        // src = e.target.result
        const img = document.createElement('img')
        img.src = e.target.result
        img.width = 224
        img.height = 224
        this.imgSrc = e.target.result
        console.log('img ======', img)
        callback(img)
      }
    },
    handleRemove (file, fileList) {
      console.log(file, fileList)
    },
    handlePreview (file) {
      console.log(file)
    },
    handleExceed (files, fileList) {
      this.$message.warning(`当前限制选择 3 个文件，本次选择了 ${files.length} 个文件，共选择了 ${files.length + fileList.length} 个文件`)
    },
    beforeRemove (file, fileList) {
      return this.$confirm(`确定移除${file.name}？`)
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- brand ------------')
    console.log('tf', tf)
    console.log('tfvis', tfvis)
    // this.addModel()
  },
  filters: {
  }
}
</script>
<style  scoped>
.box{
  width: 20vw;

}
.upload-demo{
  margin-top: 1rem;
}
</style>
