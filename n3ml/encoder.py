import numpy as np

import torch


# class PoissonEncoder:
#     # Poisson processor for encoding images
#     def __init__(self, time_interval):
#         self.time_interval = time_interval
#
#     def __call__(self, images):
#         # images.size: [b, c, h, w]
#         # spiked_images.size: [t, b, c, h, w]
#         b, c, h, w = images.size()
#         r = images.unsqueeze(0).repeat(self.time_interval, 1, 1, 1, 1) / 32.0
#         p = torch.rand(self.time_interval, b, c, h, w)
#         return (p <= r).float()
#
#
# class PoissonEncoder2:
#     def __init__(self, time_interval):
#         self.time_interval = time_interval
#
#     def __call__(self, images):
#         rate = torch.zeros(size)
#         rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)


class Simple:
    """ This is a simple version encoder

        It has to inherit base encoder for consistency.

    """
    def __init__(self, time_interval: int = 100, scale: float = 5.0) -> None:
        """

            scale에 대한 고찰
            scale < 1.0인 경우는 more deterministic 특성을 보이게 된다.
            scale > 1.0인 경우는 more stochastic 특성을 보이게 된다. (more realistic spike train)

        :param time_interval:
        :param scale:
        """
        self.time_interval = time_interval
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x.size: [1, 28, 28]
        xx = x.unsqueeze(dim=0).repeat(self.time_interval, 1, 1, 1)
        r = torch.rand([self.time_interval] + [_ for _ in x.size()], device=x.device)
        return (xx >= self.scale * r).float()


class SimplePoisson:
    # Poisson processor for encoding images
    def __init__(self, time_interval):
        self.time_interval = time_interval

    def __call__(self, images):
        # images.size: [c, h, w]
        # spiked_images.size: [t, c, h, w]
        c, h, w = images.size()
        r = images.unsqueeze(0).repeat(self.time_interval, 1, 1, 1) / 32.0
        p = torch.rand(self.time_interval, c, h, w)
        return (p <= r).float()


class Encoder:
    # language=rst
    """
    Base class for spike encodings transforms.

    Calls ``self.enc`` from the subclass and passes whatever arguments were provided.
    ``self.enc`` must be callable with ``torch.Tensor``, ``*args``, ``**kwargs``
    """

    def __init__(self, *args, **kwargs) -> None:
        self.enc_args = args
        self.enc_kwargs = kwargs

    def __call__(self, img):
        return self.enc(img, *self.enc_args, **self.enc_kwargs)


class PoissonEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable PoissonEncoder which encodes as defined in
        ``bindsnet.encoding.poisson`

        :param time: Length of Poisson spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = poisson


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)


class PopulationEncoder:
    def __init__(self, neurons, I_min, I_max, max_firing_times, not_to_fire, time_steps, beta=1.5):
        self.neurons = neurons
        self.I_min = I_min
        self.I_max = I_max
        self.max_firing_times = max_firing_times
        self.not_to_fire = not_to_fire
        self.time_steps = time_steps
        self.beta = beta
        self.mean = np.zeros((len(self.I_min), self.neurons))
        self.dev = np.zeros(len(self.I_min))

        self.mean_dev()

    def mean_dev(self):
        for n in range(self.mean.shape[0]):
            self.dev[n] = (self.I_max[n]-self.I_min[n])/(self.beta*(self.neurons-2))
            for i in range(self.mean.shape[1]):
                self.mean[n, i] = self.I_min[n]+(2*i-3)*(self.I_max[n]-self.I_min[n])/(2*(self.neurons-2))

    def density(self, x, mu, sig):
        return np.exp(-(x-mu)**2/(2*sig**2))/(np.sqrt(2*np.pi*sig**2))

    def __call__(self, x):
        """
        :param x: a list of real-values
        :return:
        """
        o = np.zeros((len(self.I_min), self.neurons))
        for n in range(self.mean.shape[0]):
            for i in range(self.mean.shape[1]):
                o[n, i] = self.density(x[n], self.mean[n, i], self.dev[n])
        for n in range(self.mean.shape[0]):
            max_density = self.density(0, 0, self.dev[n])
            for i in range(self.mean.shape[1]):
                o[n, i] = o[n, i]/max_density
        for n in range(self.mean.shape[0]):
            for i in range(self.mean.shape[1]):
                o[n, i] = int(-o[n, i] * self.max_firing_times + self.max_firing_times) * self.time_steps
                if o[n, i] >= self.not_to_fire*self.time_steps:
                    o[n, i] = -1
        return o

"""
    data_encoder = n3ml.encoder.Population(neurons=12,
                                           minimum=summary['min'],
                                           maximum=summary['max'],
                                           max_firing_time=opt.max_firing_time,
                                           not_to_fire=opt.not_to_fire,
                                           dt=opt.dt)
"""

class Population(Encoder):
    def __init__(self, neurons: int,
                 minimum: torch.Tensor,
                 maximum: torch.Tensor,
                 max_firing_time: int,
                 not_to_fire: int,
                 dt: int,
                 beta: float = 1.5) -> None:
        super().__init__()

        self.neurons = neurons
        self.minimum = minimum
        self.maximum = maximum
        self.max_firing_time = max_firing_time
        self.not_to_fire = not_to_fire
        self.dt = dt
        self.beta = beta
        self.mean = torch.zeros((minimum.size(0), neurons))
        self.dev = torch.zeros((minimum.size(0)))
        self.pi = torch.tensor(np.pi)

        self.mean_dev()

    def mean_dev(self):
        for n in range(self.mean.size(0)):
            self.dev[n] = (self.maximum[n]-self.minimum[n])/(self.beta*(self.neurons-2))
            for i in range(self.mean.size(1)):
                self.mean[n, i] = self.minimum[n]+(2*i-3)*(self.maximum[n]-self.minimum[n])/(2*(self.neurons-2))

    def density(self, x, mu, sig):
        return torch.exp(-(x-mu)**2/(2*sig**2))/(torch.sqrt(2*self.pi*sig**2))

    def run(self, x: torch.Tensor) -> None:
        o = torch.zeros(self.minimum.size(0), self.neurons)
        for n in range(self.mean.size(0)):
            for i in range(self.mean.size(1)):
                o[n, i] = self.density(x[n], self.mean[n, i], self.dev[n])
        for n in range(self.mean.size(0)):
            max_density = self.density(0, 0, self.dev[n])
            for i in range(self.mean.size(1)):
                o[n, i] /= max_density
        for n in range(self.mean.size(0)):
            for i in range(self.mean.size(1)):
                # o[n, i] = torch.round(-o[n, i] * self.max_firing_time + self.max_firing_time)
                o[n, i] = torch.round(-o[n, i] * self.max_firing_time + self.max_firing_time)
                if o[n, i] >= self.not_to_fire:
                    o[n, i] = -1
        return o
