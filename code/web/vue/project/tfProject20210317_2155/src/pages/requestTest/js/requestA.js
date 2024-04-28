
var test = require('./foobar')
console.log('test', test)
console.log('test', test.foobar)
test.foobar.constructor('22')
test.foobar.foo()
