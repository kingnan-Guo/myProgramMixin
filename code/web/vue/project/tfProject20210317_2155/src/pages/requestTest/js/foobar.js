function Foobar (params) {
  this.foo = function () {
    console.log('foo', params)
  }
}

var foobar = new Foobar()
exports.foobar = foobar
