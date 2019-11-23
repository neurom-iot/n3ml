window.onload = function() {
    Snap.plugin(function(Snap, Element, Paper, global) {
        Element.prototype.drawAtPath = function(path, timer, options) {
            var myObject = this;
            var bbox = this.getBBox(1);
            var point = {};
            var movePoint = {};
            var len = path.getTotalLength();
            var from = 0;
            var to = len;
            var drawpath = 0;
            var easing = mina.linear;
            var callback;
            var startingTransform = '';

            if (options) {
                easing = options.easing || easing;
                if (options.reverse) {
                    from = len;
                    to = 0;
                }
                if (options.drawpath) {
                    drawpath = 1;
                    path.attr({
                        fill: 'none',
                        strokeDasharray: len + ' ' + len,
                        strokeDashoffset: this.len
                    });
                }
                if (options.startingTransform) {
                    startingTransform = options.startingTransform;
                }
                callback = options.callback || function() {};
            }
        }
    });
};