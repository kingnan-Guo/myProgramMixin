<template>
  <div class="requestTest">
    <h1>requestTest</h1>
    <span>render </span>

  </div>
</template>

<script>

// import './js/foobar'
import './js/requestA'

export default {
  name: 'requestTest',
  components: {},
  data () {
    return {
      preModel: null
    }
  },
  methods: {
    addData () {
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- requestTest ------------')
  }
//   filters: {
//   }
}
</script>
