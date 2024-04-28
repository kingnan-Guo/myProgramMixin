<template>
  <div class="mobilenet">
    <h1>Pre training modelaria-rowspan</h1>
    <span>mobilenet</span>
    <!-- <img :src='imgSrc'  width='224' height='224' /> -->
    <div class="imageBox" v-html="imgSrc"></div>
    <div class="box">
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
import { IMAGENET_CLASSES } from '@/assets/resourcePage/js-ml-code/mobilenet/imagenet_classes.js'

export default {
  name: 'Mobilenet',
  data () {
    return {
      preModel: null,
      MOBILENET_MODEL_PATH: '../static/data/mobilenet/web_model/model.json',
      fileList: [],
      imgSrc: null,
      model: null
    }
  },
  methods: {
    beforeUpload (file) {
      console.log('file', file)
      // return false
      // return isJPG && isLt2M;
      this.readImg(file, (imgData) => {
        const pred = tf.tidy(() => {
          // 为了归一化 -255 但是数据 要求 （-1， 1） 所以，先减去 （255/2）再除以（）
          const input = tf.browser.fromPixels(imgData).toFloat().div(255 / 2).reshape([1, 224, 224, 3])
          console.log(input)
          // var predreshape = input.reshape([1, 224, 224, 3])
          // predreshape.print()
          // console.log(predreshape)
          return this.model.predict(input)
        })
        const index = pred.argMax(1).dataSync()[0]
        console.log('index ==', index)
        console.log('pred ==', pred)
        console.log('pred.argMax(1)', pred.argMax(1))
        console.log(IMAGENET_CLASSES[index])
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
    },
    async addModel () {
      // 加载 model 有 then 后 无reture ; model = undefined
      // const model = await tf.loadLayersModel(this.MOBILENET_MODEL_PATH).then((model) => {
      //   this.predict(val)
      // })
      await tf.loadLayersModel(this.MOBILENET_MODEL_PATH).then((model) => {
        this.predict(model)
        console.log('MOBILENET_MODEL_PATH model', model)
        this.model = model
      })
    },
    predict (model) {
      console.log('MOBILENET_MODEL_PATH model', model)
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- mobilenet ------------')
    console.log('tf', tf)
    console.log('tfvis', tfvis)
    console.log('IMAGENET_CLASSES', IMAGENET_CLASSES[0])
    console.log('MOBILENET_MODEL_PATH', this.MOBILENET_MODEL_PATH)
    this.addModel()
  },
  filters: {
  }
}
</script>
<style>
  .box{
    width: 20vw;
  }
</style>
