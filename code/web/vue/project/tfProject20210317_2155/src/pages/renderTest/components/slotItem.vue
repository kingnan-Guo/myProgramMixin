<template>
  <div class="slotItem">
    <h2>子组件</h2>
    <button>
        <slot>这就是默认值</slot>
    </button>
    <div>
        <slot name= 'two' ></slot>
        <br>
        <slot name= 'one' ></slot>
    </div>
    <div>
        <slot name= 'one' :value1='child1'> 这就是默认值1</slot>
        <slot :='child2'> 这就是默认值2 </slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'slotItem',
  components: {},
  data () {
    return {
      child: 'child data',
      child1: '数据1',
      child2: '数据2'
    }
  },
  methods: {
    addData () {
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- slotItem ------------')
  },
  filters: {
  }
}
</script>
