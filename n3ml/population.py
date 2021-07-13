from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable, Function

import matplotlib.pyplot as plt


class Population(nn.Module):
    def __init__(self):
        super().__init__()


class Input(Population):
    def __init__(self,
                 neurons: int,
                 dt: float = 1.0,
                 traces: bool = True,
                 tau_tr: float = 20.0,
                 scale_tr: float = 1.0) -> None:
        super().__init__()
        self.neurons = neurons
        self.traces = traces
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('s', torch.zeros(neurons))
        if traces:
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

    def init_param(self):
        self.s.zero_()
        if self.traces:
            self.x.zero_()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.s[:] = x
        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)
            # self.x[:] += self.scale_tr * self.s
            self.x.masked_fill_(self.s.bool(), 1)
        return self.s


class LIF(Population):
    def __init__(self,
                 neurons: int,
                 dt: float = 1.0,
                 tau_rc: float = 100.0,
                 v_th: float = -52.0,
                 rest: float = -65.0,
                 reset: float = -65.0,
                 tau_ref: float = 5.0,
                 traces: bool = True,
                 tau_tr: float = 20.0,
                 scale_tr: float = 1.0) -> None:
        super().__init__()
        self.neurons = neurons
        self.traces = traces
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        if traces:
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

    def init_param(self):
        self.v.fill_(self.rest)
        self.refrac.zero_()
        self.s.zero_()
        if self.traces:
            self.x.zero_()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest
        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt

        self.s[:] = self.v >= self.v_th

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.reset)

        # Update spike traces
        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)

            # self.x[:] += torch.scale_tr * self.s
            self.x.masked_fill_(self.s.bool(), 1)

        return self.s


class DiehlAndCook(Population):
    def __init__(self,
                 neurons: int,
                 dt: float = 1.0,
                 tau_rc: float = 100.0,
                 v_th: float = -52.0,
                 theta: float = 0.05,
                 rest: float = -65.0,
                 reset: float = -65.0,
                 tau_ref: float = 5.0,
                 traces: bool = True,
                 tau_tr: float = 20.0,
                 scale_tr: float = 1.0,
                 theta_plus: float = 0.05):
        super().__init__()
        self.neurons = neurons
        self.traces = traces
        self.reset_theta = theta
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('theta', torch.zeros(neurons))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        self.register_buffer('theta_plus', torch.tensor(theta_plus))
        if traces:
            # TODO: 초기화는 어떤 값으로 해야 하는가?
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

    def init_param(self):
        self.v.fill_(self.rest)
        self.refrac.zero_()

        self.s.zero_()

        if self.traces:
            self.x.zero_()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest
        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt

        self.s[:] = self.v >= self.v_th + self.theta

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.reset)

        # Update adaptive threshold
        self.theta += self.theta_plus * self.s

        # Update spike traces
        if self.traces:
            self.x *= torch.exp(-self.dt / self.tau_tr)

            # self.x += self.scale_tr * self.s
            self.x.masked_fill_(self.s.bool(), 1)

        return self.s


class IF1d(Population):
    def __init__(self,
                 neurons: int,
                 dt: float = 1.0,
                 leakage: float = 0.0,
                 v_th: float = 1.0,
                 tau_ref: float = 0.0,
                 rest: float = 0.0,
                 reset: float = 0.0) -> None:
        super().__init__()

        self.neurons = neurons

        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('l', torch.tensor(leakage))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))

    def run(self, x: torch.Tensor) -> None:
        self.v += self.l

        self.v += (self.refrac <= 0).float() * x
        self.refrac -= self.dt

        self.s.masked_fill_(self.v >= self.v_th, 1)

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.reset)
        self.v.masked_fill_(self.v <= self.rest, self.rest)

        return self.s


class IF2d(Population):
    def __init__(self):
        super().__init__()

    def run(self, x: torch.Tensor) -> None:
        pass
