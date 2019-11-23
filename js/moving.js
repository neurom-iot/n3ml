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
            var callback;
            var startingTransform = '';

            if (options) {
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

            Snap.animate(from, to, function(val) {
                point = path.getPointAtLength(val);
                movePoint.x = point.x - bbox.cx;
                movePoint.y = point.y - bbox.cy;
                myObject.transform(startingTransform + 't' + movePoint.x + ',' + movePoint.y + 'r' + point.alpha);
            }, timer, callback);
        }
    });

    var s = Snap(1000, 1000);

    var path = s.path('M 60 0 L 120 0 L 180 60 L 180 120 L 120 180 L 60 180 L 0 120 L 0 60 Z').attr({
        fill: 'none',
        stroke: 'red',
        opacity: 1
    });
    
    var rect = s.rect(60, 0, 20, 20).attr({fill: 'blue', opacity: 0});
    var rect2 = rect.clone();

    function drawRect(el) {
        el.drawAtPath(path, 7000, {callback: drawRect.bind(null, el)});
    };

    for (var x=0; x<1; x++) {
        setTimeout(function() {
            drawRect(rect.clone().attr({opacity: 1}))
        }, x*1000);
    }
};