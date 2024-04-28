<template>
  <div class="brand">
    <h1>Transfer Learning </h1>
    <span>brand</span>
    <!-- <img :src='imgSrc'  width='224' height='224' /> -->
    <!-- <img src='./static/data/brand/train/android-0.jpg'  width='224' height='224' /> -->
    <div class="imageBox" v-html="imgSrc"></div>
    <div class="box">
      <el-button size="small" type="primary" @click="getInputs()" >getInputs</el-button>
    </div>
  </div>
</template>

<script>

import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
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
      imageSize: 244,
      imageUrlList: []
    }
  },
  methods: {
    getInputs () {
      // for (let index = 0; index < array.length; index++) {
      //   const element = array[index]
      // }
      // this.imgSrc = this.readImgList('../static/data/brand/train/android-0.jpg')
      // console.log('this.imgSrc', this.imgSrc)
      var loadimages = []
      const requireContext = require.context('@/assets/resourcePage/js-ml-code/data/brand/train', true, /\.jpg$/)
      // 批量读取路径下的文件
      requireContext.keys().map(key => {
        // 
        const image = requireContext(key)
        // 将所有的 promis 存入 数组
        loadimages.push(this.readImgList(image))
      })
      console.log('loadimages', loadimages)
      // 处理 promise 
      Promise.all(loadimages).then((result) => {
        // 返回所有的图片对象
        console.log(result) 
      })
            
    },
    readImgList (src) {
      return new Promise(resolve => {
        const image = new Image()
        // img.crossOrigin = "anonymous";
        image.src = src
        image.width = this.imageSize
        image.height = this.imageSize
        this.imgSrc = src
        // console.log(' this.imgSrc',  this.imgSrc);
        image.onload = () => resolve(image)
      })
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- brand ------------')
    console.log('tf', tf)
    console.log('tfvis', tfvis)
    // console.log('train_box ==', train_box);
    const requireContext = require.context('@/assets/resourcePage/js-ml-code/data/brand/train', true, /\.jpg$/)
    // const req = require.contexts('/static/data/brand/train', true, /\.jpg$/)
    // console.log('req ==========', req.keys())
    requireContext.keys().map(key => {
      const mod = requireContext(key)
      // console.log('context  mod', mod)
      this.readImgList(mod)
      // this.readImgList()
    })
    // console.log('req ========', req.id)
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
