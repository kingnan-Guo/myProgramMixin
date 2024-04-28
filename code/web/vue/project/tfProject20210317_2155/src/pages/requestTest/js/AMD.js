define([
    'require',
    'dependency'
], function(require, factory) {
    // 'use strict';
    function foo(params) {
        require.test()
    }
    return {foo: foo}
});

define(function(require, exports,factory) {
    // 'use strict';
    var reqModule = require('./someModule')
    requireModule.test()
    exports.asplode = function (params) {
        
    }
});