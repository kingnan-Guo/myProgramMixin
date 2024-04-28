<template>
  <div class="renderTest">
    <h1>renderTest</h1>
    <span>render </span>
    <div>
        <items />
    </div>
    <!-- <template slot="title">
        <p>我是name为header的slot</p>
    </template> -->
    <div>
        <slotI>{{parent}}</slotI>
    </div>
    <div>
        <slotI>
            <template v-slot:one> 这是插入到one插槽的内容 </template>
            <template v-slot:two> 这是插入到two插槽的内容 </template>
        </slotI>
    </div>

    <div>
        <slotI>
            <template v-slot:one = 'slotone'>
                {{slotone.value1}}
            </template>
            <template v-slot:default = 'slottwo'>
                {{slottwo.value2}}
            </template>
        </slotI>
    </div>
  </div>
</template>

<script>
import items from './components/items'
import slotI from './components/slotItem'
export default {
  name: 'renderTest',
  components: {
    items,
    slotI
  },
  data () {
    return {
      preModel: null,
      parent: '父组件'
    //   slotone:{
    //       value1: 'value1',
    //       value2: 'value2'
    //   }
    }
  },
  methods: {
    addData () {
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- renderTest ------------')
  },
  filters: {
  }
}
</script>
