# n3ml
Neuromorphic Neural Network Machine Learning (N3ML) is designed to provide the following features.
- Construct spiking neural networks
- Simulate spiking neural networks
- Transform spiking neural networks
- Visualize spiking neural networks

N3ML is implemented by wrapping some features in PyTorch[7]. N3ML wraps `torch.nn.Module` class to track the structure of the built neural network easily and provide the compatiability with operators in PyTorch. We can construct a spiking neural network consisting of both the operators provided in N3ML and the ones in PyTorch. We can track the architecture of a spiking neural network using the methods provided by `torch.nn.Module` class. It is helpful to transform a spiking neura network to other simulators/frameworks or neuromorphic hardwares.

N3ML uses `torch.autograd.Function` to implemet surrogate derviative-based learning algorithms.

N3ML wraps `torch.Tensor` to get GPU accelerations.

### Constructing spiking neural networks

### Simulating spiking neural networks

### Transforming spiking neural networks

### Visualizing spiking neural networks

## Supporting learning algorithms
n3ml provides some learning algorithms as examples that can be simulated. If you want to simulate them, you can go to test directory.
Now, n3ml supports the following learning algorithms
1. SpikeProp [1]
2. STDP [2]
3. Soft LIF [3]
4. BP-STDP [4]
5. STBP [5]

## Installation
Now, n3ml is tested in Python 3.7.7.

### Install dependencies
If you are installing from pip, some dependencies will be installed automatially that are registered in setup.py. Also, others can be found in requirements.txt.
```
python -r requirements.txt
```
You can install n3ml using pip.
```
pip install n3ml-python
```

## GPU acceleration
We follows BindsNET[6] to supprot GPU acceleration in N3ML. Bascially, N3ML is implemented by wrapping torch.Tensor similar to BindsNET.

## How to Contribute?
- Feel free to create issues/pull-requests if you have any issues.
- Please read [CONTRIBUTING.md](CONTRIBUTING.md) if you are interested in contributing the project.

## Contributors

## References
[1] Bohte, S. M., J. N. Kok, and H. L. Poutre, Error-backpropagation in temporally encoded networks of spiking neurons. Neurocomputing, 48(1-4), 17-37 (2002)

[2] Diehl, P. U. and M. Cook, Unsupervised learning of digit recognition using spike-timing-dependent plasticity, Frontiers in computational neuroscience, 9, 99 (2015)

[3] Hunsberger, E. and C. Eliasmith, Spiking Deep Networks with LIF Neurons, arXiv preprint arXiv:1510.08829 (2015)

[4] Tavanaei, A. and A. Maida, BP-STDP: Approximating backpropagation using spike timing dependent plasticity, Neurocomputing, 330, 39-47 (2019)

[5] Wu, Y., L. Deng, G. Li, J. Zhu, and L. Shi, Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks, Frontiers in neuroscience, 12, 331 (2018)

[6] Hazan, H., D. J. Saunders, H. Khan, D. Patel, D. T. Sanghavi, H. T. Siegelmann, and R. Kozma, BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python, Frontiers in neuroinformatics, 12, 89 (2018)
