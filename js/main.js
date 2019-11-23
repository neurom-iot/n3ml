function Node(n_neurons, periods) {
    this.n_neurons = n_neurons;
    this.periods = periods;
}

function Population(n_neurons) {
    this.n_neurons = n_neurons
}

function Connection(pre, post) {
    this.pre = pre;
    this.post = post;
}

function Signal(name) {
    this.name = name;
}

window.onload = function() {
    // Now, start model representation
    var population = new Population(3);
    
    var layers = [
        {n_neurons: 2},
        {n_neurons: 3},
        {n_neurons: 2}
    ];

    // Now, start graphical processing
    Snap.plugin(function(Snap, Element, Paper, global) {
        // To visualize, we make the graph of layers
        Paper.prototype.make_graph = function(layers) {
            // ISSUE#001
            // Drawing nodes
            nodes = [
                [this.circle(300, 200, 50), this.circle(300, 350, 50)],
                [this.circle(600, 125, 50), this.circle(600, 275, 50), this.circle(600, 425, 50)],
                [this.circle(900, 200, 50), this.circle(900, 350, 50)]
            ];

            for (var i=0; i<nodes.length; i++) {
                for (var j=0; j<nodes[i].length; j++) {
                    nodes[i][j].attr({
                        fill: '#ffffff',
                        stroke: '#000000',
                        strokeWidth: 2
                    });
                }
            }

            // Drawing edges
            edges = [];

            for (var i=0; i<nodes.length-1; i++) {
                // each pair
                for (var j=0; j<nodes[i].length; j++) {
                    for (var k=0; k<nodes[i+1].length; k++) {
                        var bb1 = nodes[i][j].getBBox();
                        var bb2 = nodes[i+1][k].getBBox();

                        // TODO
                        var radian = Math.atan(bb2.cy-bb1.cy/bb2.cx-bb1.cx);

                        var path = 'M' + bb1.cx + ',' + bb1.cy + 'L' + bb2.cx + ',' + bb2.cy;

                        edges.push(this.path(path).attr({
                            fill: 'none',
                            stroke: '#000000',
                            strokeWidth: 2
                        }));
                    }
                }
            }

            graph = '';

            return graph;
        };

        Paper.prototype.make_plotbox = function(layers) {
            boxes = [
                this.rect(230, 520, 150, 100),
                this.rect(530, 520, 150, 150),
                this.rect(830, 520, 150, 100)
            ];

            for (var i=0; i<boxes.length; i++) {
                boxes[i].attr({
                    fill: '#ffffff',
                    stroke: '#000000',
                    strokeWidth: 2
                });
            }

            return boxes;
        };
    });

    var s = Snap(5000, 5000);

    var graph = s.make_graph(layers);

    var boxes = s.make_plotbox(layers);

    console.log(graph);
};