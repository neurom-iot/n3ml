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
    var population = new Population(3);
    
    var layers = [
        {n:2},
        {n:3},
        {n:2}
    ];

    var s = Snap(1000, 1000);

    
};