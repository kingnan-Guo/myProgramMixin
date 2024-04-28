<script>

export default {
  name: 'items',
  functional: true,
  data () {
    return {
      info: null
    }
  },
  methods: {
    addData () {
    }
  },
  watch: {
  },
  mounted () {
    console.log('------------- items ------------')
  },
  filters: {
  },
  render (h, context) {
    // console.log('render ==', h, 'context', context)
    var vnodes = []
    vnodes.push(<span slot='title'>{(123)}</span>)
    return vnodes
  }
}
</script>
