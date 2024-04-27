import { Base } from './BaseLayer'

class newLayer extends Base {
  constructor (value) {
    super()
    // this.value = value
  }
  __init__ () {
    console.log('')
    return 'newLayer ---- pravate init function'
  }
}
export {
  newLayer as NewLayer
}
