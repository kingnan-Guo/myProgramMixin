import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'

import Main from '@/pages/main/index'
// import home from '@/pages/home/home'
// import tensorflow from '@/pages/tensorflow/tensorflow'
import LinearRegression from '@/pages/LinearRegression/LinearRegression'
import normalization from '@/pages/normalization/normalization'
import logisticRegression from '@/pages/logisticRegression/logisticRegression'
import xor from '@/pages/xor/xor'
import iris from '@/pages/iris/iris'
import mnist from '@/pages/mnist/mnist'
import mobilenet from '@/pages/mobilenet/mobilenet'
import brand from '@/pages/brand/brand'
import renderTest from '@/pages/renderTest/renderTest'
import requestTest from '@/pages/requestTest/requestTest'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'HelloWorld',
      component: HelloWorld
    },
    {
      path: '/main', // 主屏
      name: 'Main',
      component: Main,
      children: [
        {
          path: '/linearRegression',
          name: 'linearRegression',
          component: LinearRegression
        }, {
          path: '/normalization',
          name: 'normalization',
          component: normalization
        }, {
          path: '/logisticRegression',
          name: 'logisticRegression',
          component: logisticRegression
        }, {
          path: '/xor',
          name: 'xor',
          component: xor
        }, {
          path: '/iris',
          name: 'iris',
          component: iris
        }, {
          path: '/mnist',
          name: 'mnist',
          component: mnist
        }, {
          path: '/mobilenet',
          name: 'mobilenet',
          component: mobilenet
        }, {
          path: '/brand',
          name: 'brand',
          component: brand
        }, {
          path: '/renderTest',
          name: 'renderTest',
          component: renderTest
        }, {
          path: '/requestTest',
          name: 'requestTest',
          component: requestTest
        }
      ]
    }
  ]
})
