import torch
import torch.nn as nn
import torch.autograd as autograd


class Layer(torch.nn.Module):
    def __init__(self):
        super(Layer, self).__init__()


class LIF1d(Layer):
    """
    현재 구현된 threshold와 leakage는 layer-wise scheme으로 볼 수 있다.
    그러나, neuron-wise scheme의 지원이 필요할 수 있다.
    """
    def __init__(self,
                 batch_size: int,
                 neurons: int,
                 threshold: float,
                 leakage: float,
                 reset: float = 0.0) -> None:
        super(LIF1d, self).__init__()
        self.register_buffer('th', torch.tensor(threshold))
        self.register_buffer('leak', torch.tensor(leakage))
        self.register_buffer('rst', torch.tensor(reset))
        self.register_buffer('v', torch.zeros(batch_size, neurons))
        self.register_buffer('s', torch.zeros(batch_size, neurons))

    def reset_variables(self, batch_size: int) -> None:
        self.v = torch.full(size=(batch_size, self.v.size(1)), fill_value=self.rst.item(), device=self.v.device)
        self.s = torch.zeros(size=(batch_size, self.s.size(1)), device=self.s.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.leak * self.v + x
        self.s[:] = (self.v >= self.th).float()
        self.v.masked_fill(self.v >= self.th, self.rst.item())
        return self.s


class LIF2d(Layer):
    def __init__(self,
                 batch_size: int,
                 planes: int,
                 height: int,
                 width: int,
                 threshold: float,
                 leakage: float,
                 reset: float = 0.0) -> None:
        super(LIF2d, self).__init__()
        self.register_buffer('th', torch.tensor(threshold))
        self.register_buffer('leak', torch.tensor(leakage))
        self.register_buffer('rst', torch.tensor(reset))
        self.register_buffer('v', torch.zeros(batch_size, planes, height, width))
        self.register_buffer('s', torch.zeros(batch_size, planes, height, width))

    def reset_variables(self, batch_size: int) -> None:
        self.v = torch.full(size=(batch_size, self.v.size(1), self.v.size(2), self.v.size(3)),
                            fill_value=self.rst.item(), device=self.v.device)
        self.s = torch.zeros(size=(batch_size, self.s.size(1), self.s.size(2), self.s.size(3)),
                             device=self.s.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.leak * self.v + x
        self.s[:] = (self.v >= self.th).float()
        self.v.masked_fill(self.v >= self.th, self.rst.item())
        return self.s


class SoftIF1d(nn.Module):
    def __init__(self, batch_size: int, num_features: int, threshold: float = 1.0) -> None:
        """
            v: [batch size, # features] membrane potentials
            s: [batch size, # features] spikes
        """
        super(SoftIF1d, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.threshold = torch.tensor(threshold)
        self.v = torch.zeros(size=(batch_size, num_features),
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        self.s = torch.zeros(size=(batch_size, num_features),
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        # self.v = torch.zeros(size=(batch_size, num_features))
        # self.s = torch.zeros(size=(batch_size, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x.size: [batch size, # features]

        :param x:
        :return:
        """
        self.v += x
        self.s[:] = (self.v >= self.threshold).float()
        self.v[self.v >= self.threshold] -= self.threshold
        return self.s

    def extra_repr(self) -> str:
        print('v.device: {}, v: {}'.format(self.v.device, self.v))
        print('s.device: {}, s: {}'.format(self.s.device, self.s))
        return 'batch_size={}, # features={}, threshold={}'.format(
            self.batch_size, self.num_features, self.threshold
        )

    def init_vars(self):
        self.v.fill_(0.0)
        self.s.fill_(0.0)


class SoftIF2d(nn.Module):
    def __init__(self, batch_size: int, num_channels: int, height: int, width: int, threshold: float = 1.0) -> None:
        super(SoftIF2d, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.threshold = torch.tensor(threshold)
        self.v = torch.zeros(size=(batch_size, num_channels, height, width),
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        self.s = torch.zeros(size=(batch_size, num_channels, height, width),
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        # self.v = torch.zeros(size=(batch_size, num_channels, height, width))
        # self.s = torch.zeros(size=(batch_size, num_channels, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [batch size, # channels, height, width] input currents
            v: [batch size, # channels, height, width] membrane potentials
            s: [batch size, # channels, height, width] spikes
        :param x:
        :return:
        """
        self.v += x
        self.s[:] = (self.v >= self.threshold).float()
        self.v[self.v >= self.threshold] -= self.threshold
        return self.s

    def extra_repr(self) -> str:
        print('v.device: {}, v: {}'.format(self.v.device, self.v))
        print('s.device: {}, s: {}'.format(self.s.device, self.s))
        return 'batch_size={}, # channels={}, height={}, width={}, threshold={}'.format(
            self.batch_size, self.num_channels, self.height, self.width, self.threshold
        )

    def init_vars(self):
        self.v.fill_(0.0)
        self.s.fill_(0.0)


def softplus(x, sigma=1.):
    y = torch.true_divide(x, sigma)
    z = x.clone().float()
    z[y < 34.0] = sigma * torch.log1p(torch.exp(y[y < 34.0]))
    return z


def lif_j(j, tau_ref, tau_rc, amplitude=1.):
    j = torch.true_divide(1., j)
    j = torch.log1p(j)
    j = tau_ref + tau_rc * j
    j = torch.true_divide(amplitude, j)
    return j


class _SoftLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain, bias, sigma, v_th, tau_ref, tau_rc, amplitude):
        ctx.save_for_backward(x, gain, bias, sigma, v_th, tau_ref, tau_rc, amplitude)
        # j = gain * x + bias - v_th
        j = gain * x
        j = softplus(j, sigma)
        o = torch.zeros_like(j)
        o[j > 0] = lif_j(j[j > 0], tau_ref, tau_rc, amplitude)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        x, gain, bias, sigma, v_th, tau_ref, tau_rc, amplitude = ctx.saved_tensors
        # y = gain * x + bias - v_th  # TODO: 1이 v_th=1로 했기 때문에 1인 건가? 아니면 다른 것에 의한 건가?
        y = gain * x
        j = softplus(y, sigma)
        yy = y[j > 1e-15]
        jj = j[j > 1e-15]
        vv = lif_j(jj, tau_ref, tau_rc, amplitude)
        d = torch.zeros_like(j)
        d[j > 1e-15] = torch.true_divide((gain * tau_rc * vv * vv),
                                         (amplitude * jj * (jj + 1) * (1 + torch.exp(torch.true_divide(-yy, sigma)))))
        grad_input = grad_output * d
        return grad_input, None, None, None, None, None, None, None


class SoftLIF(Layer):
    def __init__(self, gain=1., bias=0., sigma=0.02, v_th=1., tau_ref=0.001, tau_rc=0.05, amplitude=1.):
        super().__init__()
        self.gain = torch.autograd.Variable(torch.tensor(gain), requires_grad=False)
        self.bias = torch.autograd.Variable(torch.tensor(bias), requires_grad=False)
        self.sigma = torch.autograd.Variable(torch.tensor(sigma), requires_grad=False)
        self.v_th = torch.autograd.Variable(torch.tensor(v_th), requires_grad=False)
        self.tau_ref = torch.autograd.Variable(torch.tensor(tau_ref), requires_grad=False)
        self.tau_rc = torch.autograd.Variable(torch.tensor(tau_rc), requires_grad=False)
        self.amplitude = torch.autograd.Variable(torch.tensor(amplitude), requires_grad=False)

    def forward(self, x):
        return _SoftLIF.apply(x, self.gain, self.bias, self.sigma, self.v_th, self.tau_ref, self.tau_rc, self.amplitude)


class _Wu(torch.autograd.Function):
    threshold = 0.5
    alpha = 0.5

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        return x.gt(_Wu.threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(x - _Wu.threshold) < _Wu.alpha
        return grad_input * temp.float()


class Wu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _Wu.apply(x)


class _Wu_alternative(autograd.Function):
    @staticmethod
    def forward(ctx, voltages, threshold, alpha):
        ctx.save_for_backward(voltages, threshold, alpha)
        return voltages.gt(threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(x - threshold) < alpha
        grad_output = grad_input * temp.float()
        return grad_output, None, None


class Wu2d(nn.Module):
    def __init__(self,
                batch_size: int,
                planes: int,
                height: int,
                width: int,
                leakage: float = 0.2,
                threshold: float = 0.5,
                alpha: float = 0.5) -> None:
        super(Wu2d, self).__init__()
        self.batch_size = batch_size
        self.planes = planes
        self.height = height
        self.width = width
        self.leakage = leakage
        self.threshold = torch.tensor(threshold)
        self.alpha = torch.tensor(alpha)

    def forward(self, x, v, s):
        # self.v *= self.leakage * (1.0 - self.s)
        # self.v += x
        v = v * self.leakage * (1.0 - s) + x
        s = _Wu_alternative.apply(v, self.threshold, self.alpha)
        return s

    def extra_repr(self) -> str:
        return 'batch_size={}, planes={}, height={}, width={}, leakage={}, threshold={}, alpha={}'.format(
            self.batch_size, self.planes, self.height, self.width, self.leakage, self.threshold, self.alpha)


class Wu1d(nn.Module):
    def __init__(self,
                batch_size: int,
                neurons: int,
                leakage: float = 0.2,
                threshold: float = 0.5,
                alpha: float = 0.5) -> None:
        super(Wu1d, self).__init__()
        self.batch_size = batch_size
        self.neurons = neurons
        self.leakage = leakage
        self.threshold = torch.tensor(threshold)
        self.alpha = torch.tensor(alpha)

    def forward(self, x, v, s):
        v = v * self.leakage * (1.0 - s) + x
        s = _Wu_alternative.apply(v, self.threshold, self.alpha)
        return s

    def extra_repr(self) -> str:
        return 'batch_size={}, neurons={}, leakage={}, threshold={}, alpha={}'.format(
            self.batch_size, self.neurons, self.leakage, self.threshold, self.alpha)


class Bohte(Layer):
    def __init__(self,
                 in_neurons: int,
                 out_neurons: int,
                 delays: int = 16,
                 threshold: float = 1.0,
                 time_constant: float = 5.0) -> None:
        super().__init__()

        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.delays = delays

        self.register_buffer('d', torch.zeros(delays))
        self.register_buffer('v', torch.zeros(out_neurons))
        self.register_buffer('v_th', torch.tensor(threshold))
        self.register_buffer('tau_rc', torch.tensor(time_constant))
        self.register_buffer('w', torch.zeros((out_neurons, in_neurons, delays)))
        self.register_buffer('s', torch.zeros(out_neurons))

    def initialize(self, delay=True) -> None:
        if delay:
            self.d[:] = (torch.rand(self.delays) * 10).int()
            # voltage는 초기화할 필요가 없다.
            # [0.02, 0.1]
            self.w[:] = torch.rand((self.out_neurons, self.in_neurons, self.delays)) * 0.08 + 0.02
        self.s.fill_(-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Step 1. Compute spike response y
        y = self.response(t, x, self.d)

        # Step 2. Compute voltage v
        yy = y.unsqueeze(0).repeat(self.out_neurons, 1, 1)
        self.v[:] = (self.w * yy).sum(dim=(1, 2))

        # Step 3. Compute spike time t
        # Note this is a single spike case
        self.s[torch.logical_and(self.s < 0, self.v >= self.v_th)] = t

        return self.s

    def response(self, t: torch.Tensor, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # t: 0-dimensional tensor
        # x: 1-dimensional tensor
        # d: 1-dimensional tensor
        xx = x.unsqueeze(1).repeat(1, self.delays)
        dd = d.unsqueeze(0).repeat(self.in_neurons, 1)
        tt = t - xx - dd
        o = torch.zeros((self.in_neurons, self.delays))
        o[torch.logical_and(xx != -1, tt >= 0)] = (tt * torch.exp(1 - tt / self.tau_rc) / self.tau_rc)[torch.logical_and(xx != -1, tt >= 0)]
        return o


class TravanaeiAndMaida(Layer):
    """ Now, this layer only supports fully-connected case.

        In future, we will add additional functionalities
        1. batch processing (now, batch size is always 1)
        2. convolution processing
    """
    def __init__(self,
                 in_neurons: int,
                 out_neurons: int,
                 threshold: float = 1.0,
                 reset: float = 0.0) -> None:
        super().__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.register_buffer('th', torch.tensor(threshold))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('u', torch.zeros(out_neurons))
        self.register_buffer('o', torch.zeros(out_neurons))
        self.register_buffer('w', torch.zeros(out_neurons, in_neurons))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        self.u += torch.matmul(self.w, o)
        self.o[:] = self.u >= self.th
        self.u.masked_fill_(self.u >= self.th, self.reset)
        return self.o

    def reset_variables(self, u: bool = True, w: bool = True):
        if u:
            self.u.fill_(self.reset)
        if w:
            self.w[:] = torch.normal(0, 1, self.w.size())


# class _TravanaeiAndMaida(autograd.Function):
#     @staticmethod
#     def forward(ctx, o, u, th, w):
#         ctx.save_for_backward(o, u, th, w)
#         u += torch.matmul(w, o)
#         oo = torch.zeros_like(u)
#         oo.masked_fill_(u >= th, 1)
#         return oo
#
#     @staticmethod
#     def backward(ctx, output_grad):
#         o, u, th, w = ctx.saved_tensors
#         doodw = None
#         doodo = None
#         return torch.tensor(0)
#
#
# class TravanaeiAndMaida(Layer):
#     """ Now, this layer only supports fully-connected case.
#
#         In future, we will add additional functionalities
#         1. batch processing (now, batch size is always 1)
#         2. convolution processing
#     """
#     def __init__(self,
#                  in_neurons: int,
#                  out_neurons: int,
#                  threshold: float = 1.0) -> None:
#         super().__init__()
#         self.in_neurons = in_neurons
#         self.out_neurons = out_neurons
#         self.register_buffer('u', torch.zeros(out_neurons))
#         self.register_buffer('th', torch.tensor(threshold))
#         self.register_parameter('w', nn.Parameter(torch.zeros(out_neurons, in_neurons)))
#
#     def forward(self, o: torch.Tensor) -> torch.Tensor:
#         return _TravanaeiAndMaida.apply(o, self.u, self.th, self.w)


class Conv2d(torch.nn.Module):
    def __init__(self, in_planes, planes, width, height, kernel_size, time_interval, stride=1, bias=False):
        super().__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.time_interval = time_interval
        self.stride = stride
        self.bias = bias

        self.f = nn.Conv2d(self.in_planes, self.planes, self.kernel_size, stride=self.stride, bias=self.bias)
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width)).to('cuda:0')

    def forward(self, t, x):
        self.s[t] = self.f(x[t])
        return self.s


class AvgPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride, planes, width, height, time_interval):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.planes = planes
        self.width = width
        self.height = height
        self.time_interval = time_interval

        self.f = nn.AvgPool2d(self.kernel_size, self.stride)
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width)).to('cuda:0')

    def forward(self, t, x):
        self.s[t] = self.f(x[t])
        return self.s


class Linear(torch.nn.Module):
    def __init__(self, in_neurons, neurons, time_interval, bias=False):
        super().__init__()

        self.in_neurons = in_neurons
        self.neurons = neurons
        self.time_interval = time_interval
        self.bias = bias

        self.f = nn.Linear(self.in_neurons, self.neurons, self.bias)
        self.s = torch.zeros((self.time_interval, 1, self.neurons)).to('cuda:0')

    def forward(self, t, x):
        self.s[t] = self.f(x[t])
        return self.s


class BatchIF1d(Layer):
    def __init__(self, neurons: int, batch_size: int, threshold: float, reset: float) -> None:
        super(BatchIF1d, self).__init__()
        self.neurons = neurons
        self.batch_size = batch_size
        self.register_buffer('voltage', torch.zeros(batch_size, neurons))
        self.register_buffer('spike', torch.zeros(batch_size, neurons))
        self.register_buffer('threshold', torch.tensor(threshold))
        self.register_buffer('reset', torch.tensor(reset))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch size, neurons]
        """
        self.voltage += x
        self.spike[:] = (self.voltage > self.threshold).float()
        self.voltage.masked_fill_(self.voltage >= self.threshold, self.reset)
        return self.spike


class BatchIF2d(Layer):
    def __init__(self, planes: int, height: int, width: int, batch_size: int, threshold: float, reset: float) -> None:
        super(BatchIF2d, self).__init__()
        self.planes = planes
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.register_buffer('voltage', torch.zeros(batch_size, planes, height, width))
        self.register_buffer('spike', torch.zeros(batch_size, planes, height, width))
        self.register_buffer('threshold', torch.tensor(threshold))
        self.register_buffer('reset', torch.tensor(reset))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch size, channels, height, width]
        """
        self.voltage += x
        self.spike[:] = (self.voltage > self.threshold).float()
        self.voltage.masked_fill_(self.voltage >= self.threshold, self.reset)
        return self.spike


class IF1d(torch.nn.Module):
    def __init__(self, neurons, time_interval, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        super().__init__()

        self.neurons = neurons
        self.time_interval = time_interval
        self.v_leak = leak
        self.v_th = threshold
        self.v_reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.v_th
        else:
            self.v_min = self.v_reset

        self.v = torch.zeros(self.neurons).to('cuda:0')
        # Now, batch_size is always 1
        self.s = torch.zeros((self.time_interval, 1, self.neurons)).to('cuda:0')

    def forward(self, t, x):
        # t: scalar
        # x.size: [time_interval, batch_size=1, neurons]
        # s.size: [time_interval, batch_size=1, neurons]
        self.v += self.v_leak + x[t, 0]
        self.s[t, 0, self.v >= self.v_th] = 1
        self.v[self.v >= self.v_th] = self.v_reset
        self.v[self.v < self.v_min] = self.v_min
        return self.s


class IF2d(torch.nn.Module):
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

        self.v = torch.zeros((self.planes, self.height, self.width)).to('cuda:0')
        # Now, batch_size is always 1
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width)).to('cuda:0')

    def forward(self, t, x):
        # t: scalar
        # x.size: [time_interval, batch_size=1, channels, height, width]
        # s.size: [time_interval, batch_size=1, channels, height, width]
        self.v += self.v_leak + x[t, 0]
        self.s[t, 0, self.v >= self.v_th] = 1
        self.v[self.v >= self.v_th] = self.v_reset
        self.v[self.v < self.v_min] = self.v_min
        return self.s
