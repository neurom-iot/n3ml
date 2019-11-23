window.onload = function() {
    Snap.plugin(function(Snap, Element, Paper, global) {
        Paper.prototype.connection = function(conn) {
            var bb1 = conn['from'].getBBox();
            var bb2 = conn['to'].getBBox();
            
            var path = 'M' + bb1.cx + ',' + bb1.cy + 'L' + bb2.cx + ',' + bb2.cy;

            conn['line'].attr({path: path});
        };

        Paper.prototype.make_line = function(obj1, obj2) {
            var bb1 = obj1.getBBox();
            var bb2 = obj2.getBBox();

            var path = 'M' + bb1.cx + ',' + bb1.cy + 'L' + bb2.cx + ',' + bb2.cy;

            return this.path(path).attr({stroke: 'black', fill: 'none'});
        };
    });

    var s = Snap(1000, 1000);

    var dragger = function() {
        this.data('ox', this.type == 'rect' ? this.attr('x') : this.attr('cx'));
        this.data('oy', this.type == 'rect' ? this.attr('y') : this.attr('cy'));
        this.animate({'fill-opacity': .2}, 500);
    };
    var move = function(dx, dy) {
        var att = this.type == 'rect' ? {x: parseInt(this.data('ox')) + dx, y: parseInt(this.data('oy')) + dy} : {cx: parseInt(this.data('ox')) + dx, cy: parseInt(this.data('oy')) + dy};
        this.attr(att);

        for (var i=0; i<connections.length; i++) {
            s.connection(connections[i]);
        }
    };
    var up = function() {
        this.animate({'fill-opacity': 0}, 500);
    };

    shapes = [
        s.ellipse(190, 100, 30, 20),
        s.rect(290, 80, 60, 40),
        s.rect(290, 180, 60, 40),
        s.ellipse(450, 100, 20, 20)
    ];
    connections = [];
    lines = [];

    for (var i=0; i<shapes.length; i++) {
        shapes[i].attr({stroke: 'red', 'fill-opacity': 0, 'stroke-width': 2, cursor: 'move'});
        shapes[i].drag(move, dragger, up);
    }

    lines.push(
        {from: shapes[0], to: shapes[1], line: s.make_line(shapes[0], shapes[1])},
        {from: shapes[1], to: shapes[2], line: s.make_line(shapes[1], shapes[2])},
        {from: shapes[1], to: shapes[3], line: s.make_line(shapes[1], shapes[3])}
    );

    for (var i=0; i<lines.length; i++) {
        connections.push(lines[i]);
    }
};