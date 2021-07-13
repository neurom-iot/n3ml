import numpy as np

import torch


class SpikeGenerator:
    def __init__(self, c=0.3):
        self.c = c

    def __call__(self, x):
        r = np.random.uniform(0, 1, x.shape)
        f = np.zeros(x.shape)
        f[self.c*x > r] = 1
        return f


class CIFAR10_SpikeGenerator:
    def __init__(self, planes, height, width, time_interval, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        super().__init__()
        self.planes = planes
        self.height = height
        self.width = width
        self.time_interval = time_interval
        self.v_leak = leak
        self.v_th = threshold
        self.v_reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.v_th
        else:
            self.v_min = self.v_reset

        self.v = torch.zeros((self.planes, self.height, self.width))
        # Now, batch_size is always 1
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width))

    def __call__(self, t, x):
        # t: scalar
        # x.size: [batch_size=1, channels, height, width]
        # s.size: [time_interval, batch_size=1, channels, height, width]
        self.v += x[0] + self.v_leak
        self.s[t, 0, self.v >= self.v_th] = 1
        self.v[self.v >= self.v_th] = self.v_reset
        self.v[self.v < self.v_min] = self.v_min
        return self.s

    def cuda(self):
        self.v = self.v.to('cuda:0')
        self.s = self.s.to('cuda:0')


class _CIFAR10_SpikeGenerator:
    def __init__(self, channel, height, width, threshold=1):
        self.channel = channel
        self.height = height
        self.width = width
        self.th = threshold
        self.v_min = -10*self.th

        self.v = np.zeros((channel, height, width))

    def __call__(self, x):
        # x.size: [batch_size=1, channels, height, width]
        print("x.size: {}".format(x.size()))
        assert self.v.shape == x.shape
        self.v[:] = self.v + x
        f = np.zeros(self.v.shape)
        f[self.v >= self.th] = 1
        self.v[self.v >= self.th] = self.v[self.v >= self.th] - self.th
        self.v[self.v < self.v_min] = self.v_min
        return f


class SpikeCounter:
    def __init__(self):
        pass

    def __call__(self, x):
        o = np.zeros(x[0].shape[0])
        for v in x:
            o += v
        return o