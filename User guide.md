# User Manual 

This manual aims to provide primary steps to construct and train the spiking neural networks 
with different training algorithms in n3ml package. First we list training 
approaches for SNN. Next, for each training algorithm belong to the corresponding approach, we
will provide a detail description and an implementation procedure to train SNNs with n3ml 
package. 



## SNN training approaches 
To train SNNs, the training algorithms can be categorized into three main approaches:
 biological-based learning rules, approximation of spike-based backpropagation, 
 and ANN-SNN conversion methodologies. 
 
### Biological-based learning approach
Inspired by the bio-neural system, learning rules in this approach attempt to train 
 SNN by modifying the synaptic strength based on local learning rules (STDP, R-STDP) in an 
 unsupervised/semi-supervised manner.   
####Objective

Pros: Biologically plausible

Cos: Low accuracy compared with the remaining training appproachs.

#### Synaptic Time Dependent Plasticity (STDP)

##### Description
Spike-timing-dependent-plasticity (STDP) [] trains spiking neural networks by 
adjusting the connection weight between every particular pair of pre and 
postsynaptic neurons based on the relative timing of output and input spikes.
The update rule is fomulated as follows

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W&space;=&space;\alpha(x_{pre}-x_{tar})(w_{max}-w)^{\mu}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W&space;=&space;\alpha(x_{pre}-x_{tar})(w_{max}-w)^{\mu}" title="\Delta W = \alpha(x_{pre}-x_{tar})(w_{max}-w)^{\mu}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W" title="\Delta W" /></a>
 is the weight change over time step, <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a>
 is the learning rate, <a href="https://www.codecogs.com/eqnedit.php?latex=w_{max}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{max}" title="w_{max}" /></a>
is the maximum weight and <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a>
determine the dependence of the update on the previous weight. 
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{tar}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{tar}" title="x_{tar}" /></a>
and <a href="https://www.codecogs.com/eqnedit.php?latex=x_{pre}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{pre}" title="x_{pre}" /></a>
model the history trace of postsynaptic and presynaptic spikes respectively.     

##### Implementation

1. Define 

      

   
  
 
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

## PyPI
```
python setup.py bdist_wheel
```

```
python -m twine upload dist/*
```

## Contributors

## References
[1] Bohte, S. M., J. N. Kok, and H. L. Poutre, Error-backpropagation in temporally encoded networks of spiking neurons. Neurocomputing, 48(1-4), 17-37 (2002)

[2] Diehl, P. U. and M. Cook, Unsupervised learning of digit recognition using spike-timing-dependent plasticity, Frontiers in computational neuroscience, 9, 99 (2015)

[3] Hunsberger, E. and C. Eliasmith, Spiking Deep Networks with LIF Neurons, arXiv preprint arXiv:1510.08829 (2015)

[4] Tavanaei, A. and A. Maida, BP-STDP: Approximating backpropagation using spike timing dependent plasticity, Neurocomputing, 330, 39-47 (2019)

[5] Wu, Y., L. Deng, G. Li, J. Zhu, and L. Shi, Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks, Frontiers in neuroscience, 12, 331 (2018)

[6] Hazan, H., D. J. Saunders, H. Khan, D. Patel, D. T. Sanghavi, H. T. Siegelmann, and R. Kozma, BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python, Frontiers in neuroinformatics, 12, 89 (2018)
