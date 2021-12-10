# User Manual 

This manual aims to provide primary steps to construct and train the spiking neural networks 
with different training algorithms. First we list training approaches for SNN. Next, for each training algorithm belong to the corresponding approach, we
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
#### Objective

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

To train the spiking neuron network in [] with STDP algorithm on MNIST task:

###### Step1: Prepare dataset:
Using Pytorch wrapping to load MNIST dataset.

```
import torchvision
from torchvision.transforms import transforms

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        opt.data,
        train=True,
        transform=torchvision.transforms.Compose([
            transforms.ToTensor(), transforms.Lambda(lambda x: x * 32 * 4)])),
    batch_size=opt.batch_size,
    shuffle=True)
```
 ###### Step2: Define encoding method:
Encoding is the first step in the whole SNN training process. It's responsible for 
converting input pixel intensity into a binary sequence of spike events before feeding 
into SNN. Specifically, here input intensities are converted into Poisson spike trains 
whose average firing rate is proportional to the pixel intensity ( the higher the pixel 
intensity, the higher the spike count).
   
```
from n3ml.encoder import PoissonEncoder
# Define an encoder to generate spike train for an image
encoder = PoissonEncoder(opt.time_interval)
```

###### Step3: Define SNN model:

In n3ml, the SNN model in [] is available for used. Here we define the model as follows 

```
from n3ml.model import DiehlAndCook2015
# Define a model
model = DiehlAndCook2015(neurons=100)
```

###### Step4: Training the defined model with STDP algorithm:

```
for epoch in range(opt.num_epochs):
    start = time.time()
    for step, (images, labels) in enumerate(train_loader):
        # Initialize a model
        model.init_param()

        # Encode images into spiked_images 
        images = images.view(1, 28, 28)
        spiked_images = encoder(images)
        spiked_images = spiked_images.view(opt.time_interval, -1)
        # spiked_images = spiked_images.cuda()

        # Train a model
        for t in range(opt.time_interval):
            
            # feed forward to SNN
            model.run({'inp': spiked_images[t]})

            # Update weights using STDP learning rule
            model.update()

        # Normalize weights
        model.normalize()
        
        #Observe how the weights are changing during training process
        w = model.xe.w.detach().cpu().numpy()
        fig, mat = plot(fig, mat, w)       
```
###### Step5: Putting them together:

A completed sample is now provided in test directory. 

To train SNN in [] with STDP, please run the following file:
 ```
test/test_stdp.py
```

To test the trained SNN with STDP, please run the following files in order :
 ```
test/test_stdp_assign.py
test/test_stdp_infer.py
```      
