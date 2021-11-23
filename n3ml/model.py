import torch
import torch.nn as nn
import torch.distributions.uniform

from n3ml.layer import IF1d, IF2d, Conv2d, AvgPool2d, Linear, Bohte, TravanaeiAndMaida

from n3ml.network import Network
import n3ml.layer
import n3ml.population
import n3ml.connection
import n3ml.learning
from n3ml.layer import SoftLIF, LIF1d, LIF2d


class Voelker2015(n3ml.network.Network):
    def __init__(self,
                 neurons: int,
                 input_size: int,
                 output_size: int,
                 neuron_type: str,
                 dt: float):
        """
            neuron_type에 대해서
            지금은 입력된 neuron_type에 관계없이 n3ml.population.LIF를 사용하고 있습니다.

            dt에 대해서
            dt 입력 단위는 ms가 되어야 합니다. 예를 들어, 1ms를 입력한다면 1을 입력해야 하고
            100ms를 입력하고 싶다면 100을 입력해야 합니다. 0.001로 입력한다면 이는 0.001s로
            간주되기 때문에 단위를 ms로 수정하여 입력해야 합니다.
        """
        super().__init__()
        self.neuron_type = neuron_type
        self.dt = dt
        self.add_component('pop', n3ml.population.NEF(neurons=neurons,
                                                      input_size=input_size,
                                                      output_size=output_size,
                                                      neuron_type=n3ml.population.LIF))


    def init_vars(self) -> None:
        for p in self.population.values():
            p.init_vars()

    def init_params(self) -> None:
        for p in self.population.values():
            p.init_params()


class Wu2018(Network):
    def __init__(self, batch_size: int) -> None:
        super(Wu2018, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32,  kernel_size=3, stride=1, padding=1)
        self.conv1_lif = n3ml.layer.Wu()
        self.apool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_lif = n3ml.layer.Wu()
        self.apool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc1_lif = n3ml.layer.Wu()
        self.fc2 = nn.Linear(128, 10)
        self.fc2_lif = n3ml.layer.Wu()

    def mem_update(self, ops, x, mem, spike, wu):
        mem = mem * 0.2 * (1. - spike) + ops(x)
        spike = wu(mem)
        return mem, spike

    def forward(self, images: torch.Tensor, num_steps: int) -> torch.Tensor:
        c1_mem = c1_spike = torch.zeros(self.batch_size, 32, 28, 28, device=images.device)
        c2_mem = c2_spike = torch.zeros(self.batch_size, 32, 14, 14, device=images.device)

        h1_mem = h1_spike = torch.zeros(self.batch_size, 128, device=images.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, 10, device=images.device)

        for time in range(num_steps):
            x = (images > torch.rand(images.size(), device=images.device)).float()

            c1_mem, c1_spike = self.mem_update(self.conv1, x.float(), c1_mem, c1_spike, self.conv1_lif)
            x = self.apool1(c1_spike)

            c2_mem, c2_spike = self.mem_update(self.conv2, x, c2_mem, c2_spike, self.conv2_lif)
            x = self.apool2(c2_spike)

            x = self.flatten(x)

            h1_mem, h1_spike = self.mem_update(self.fc1, x, h1_mem, h1_spike, self.fc1_lif)
            h2_mem, h2_spike = self.mem_update(self.fc2, h1_spike, h2_mem, h2_spike, self.fc2_lif)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / num_steps

        return outputs


class Ponulak2005(n3ml.network.Network):
    def __init__(self,
                 neurons: int = 800,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.neurons = neurons
        self.num_classes = num_classes
        self.add_component('input', n3ml.population.Input(1*28*28,
                                                          traces=False))
        self.add_component('hidden', n3ml.population.LIF(neurons,
                                                         tau_ref=2.0,
                                                         traces=False,
                                                         rest=0.0,
                                                         reset=0.0,
                                                         v_th=1.0,
                                                         tau_rc=10.0))
        self.add_component('output', n3ml.population.LIF(num_classes,
                                                         tau_ref=2.0,
                                                         traces=False,
                                                         rest=0.0,
                                                         reset=0.0,
                                                         v_th=1.0,
                                                         tau_rc=10.0))
        self.add_component('ih', n3ml.connection.Synapse(self.input, self.hidden))
        self.add_component('ho', n3ml.connection.Synapse(self.hidden, self.output))

    def reset_parameters(self):
        for synapse in self.connection.values():
            synapse.w[:] = torch.rand_like(synapse.w) - 0.5


class DiehlAndCook2015(n3ml.network.Network):
    def __init__(self, neurons: int = 100):
        super().__init__()
        self.neurons = neurons
        self.add_component('inp', n3ml.population.Input(1*28*28,
                                                        traces=True,
                                                        tau_tr=20.0))
        self.add_component('exc', n3ml.population.DiehlAndCook(neurons,
                                                               traces=True,
                                                               rest=-65.0,
                                                               reset=-60.0,
                                                               v_th=-52.0,
                                                               tau_ref=5.0,
                                                               tau_rc=100.0,
                                                               tau_tr=20.0))
        self.add_component('inh', n3ml.population.LIF(neurons,
                                                      traces=False,
                                                      rest=-60.0,
                                                      reset=-45.0,
                                                      v_th=-40.0,
                                                      tau_rc=10.0,
                                                      tau_ref=2.0,
                                                      tau_tr=20.0))
        self.add_component('xe', n3ml.connection.LinearSynapse(self.inp,
                                                               self.exc,
                                                               alpha=78.4,
                                                               learning_rule=n3ml.learning.PostPre,
                                                               initializer=torch.distributions.uniform.Uniform(0, 0.3)))
        self.add_component('ei', n3ml.connection.LinearSynapse(self.exc,
                                                               self.inh,
                                                               w_min=0.0,
                                                               w_max=22.5))
        self.add_component('ie', n3ml.connection.LinearSynapse(self.inh,
                                                               self.exc,
                                                               w_min=-120.0,
                                                               w_max=0.0))

        # Initialize synaptic weight for each synapse
        self.xe.init()
        self.ei.w[:] = torch.diagflat(torch.ones_like(self.ei.w)[0] * 22.5)
        self.ie.w[:] = (torch.ones_like(self.ie.w) * -120.0).fill_diagonal_(0.0)


class DiehlAndCook2015Infer(n3ml.network.Network):
    def __init__(self, neurons: int = 100):
        super().__init__()
        self.neurons = neurons
        self.add_component('inp', n3ml.population.Input(1*28*28,
                                                        traces=True,
                                                        tau_tr=20.0))
        self.add_component('exc', n3ml.population.DiehlAndCook(neurons,
                                                               traces=True,
                                                               rest=-65.0,
                                                               reset=-60.0,
                                                               v_th=-52.0,
                                                               tau_ref=5.0,
                                                               tau_rc=100.0,
                                                               tau_tr=20.0,
                                                               fix=True))
        self.add_component('inh', n3ml.population.LIF(neurons,
                                                      traces=False,
                                                      rest=-60.0,
                                                      reset=-45.0,
                                                      v_th=-40.0,
                                                      tau_rc=10.0,
                                                      tau_ref=2.0,
                                                      tau_tr=20.0))
        self.add_component('xe', n3ml.connection.LinearSynapse(self.inp,
                                                               self.exc,
                                                               alpha=78.4,
                                                               initializer=torch.distributions.uniform.Uniform(0, 0.3)))
        self.add_component('ei', n3ml.connection.LinearSynapse(self.exc,
                                                               self.inh,
                                                               w_min=0.0,
                                                               w_max=22.5))
        self.add_component('ie', n3ml.connection.LinearSynapse(self.inh,
                                                               self.exc,
                                                               w_min=-120.0,
                                                               w_max=0.0))

        # Initialize synaptic weight for each synapse
        self.xe.init()
        self.ei.w[:] = torch.diagflat(torch.ones_like(self.ei.w)[0] * 22.5)
        self.ie.w[:] = (torch.ones_like(self.ie.w) * -120.0).fill_diagonal_(0.0)


class Hunsberger2015(n3ml.network.Network):
    def __init__(self, amplitude, tau_ref, tau_rc, gain, sigma, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 1024, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.Linear(1024, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x


class Hunsberger2015_ANN(Network):
    def __init__(self):
        super(Hunsberger2015_ANN, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.named_children():
            x = m[1](x)
        return x


class Hunsberger2015_SNN(Network):
    def __init__(self, ann: Hunsberger2015_ANN, fake_images: torch.Tensor) -> None:
        super(Hunsberger2015_SNN, self).__init__()
        # Get the output's size of a conv. layer or a fc. layer to generate
        # spiking neuron model.
        shapes = {}
        fake_x = fake_images
        for m in ann.named_modules():
            fake_x = m[1](fake_x)
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                shapes[m[0]] = fake_x.size()
        # Construct a SNN based on an ANN.
        for m in ann.named_modules():
            if not isinstance(m[1], SoftLIF):
                self.add_module(m[0], m[1])
                if isinstance(m[1], nn.Conv2d):
                    self.add_module('{}.LIF2d'.format(m[0]), LIF2d(plains=shapes[m[0]][1],
                                                                   height=shapes[m[0]][2],
                                                                   width=shapes[m[0]][3]))
                elif isinstance(m[1], nn.Linear):
                    self.add_module('{}.LIF1d'.format(m[0]), LIF1d(neurons=shapes[m[0]][1]))

    def reset_variables(self, batch_size: int) -> None:
        for m in self.named_modules():
            if isinstance(m[1], LIF1d) or isinstance(m[1], LIF2d):
                m[1].reset_variables(batch_size=batch_size)

    def forward(self, images: torch.Tensor, num_steps: int) -> torch.Tensor:
        for t in range(num_steps):
            x = None  # TODO: encode 'x'
            for m in self.named_children():
                x = m[1](x)
        o = x
        return o


class Bohte2002(n3ml.network.Network):
    def __init__(self,
                 in_neurons: int = 50,
                 hid_neurons: int = 10,
                 out_neurons: int = 3,
                 delays: int = 16,
                 threshold: float = 1.0,
                 time_constant: float = 7.0) -> None:
        super(Bohte2002, self).__init__()
        self.in_neurons = in_neurons
        self.hid_neurons = hid_neurons
        self.out_neurons = out_neurons
        self.delays = delays
        self.threshold = threshold
        self.time_constant = time_constant
        self.add_component('fc1', Bohte(in_neurons,
                                        hid_neurons,
                                        delays=delays,
                                        threshold=threshold,
                                        time_constant=time_constant))
        self.add_component('fc2', Bohte(hid_neurons,
                                        out_neurons,
                                        delays=delays,
                                        threshold=threshold,
                                        time_constant=time_constant))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(t, x)
        x = self.fc2(t, x)
        return x


class TravanaeiAndMaida2017(n3ml.network.Network):
    def __init__(self,
                 num_classes: int = 10,
                 hidden_neurons: int = 100) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_neurons = hidden_neurons
        self.add_component('fc1', TravanaeiAndMaida(in_neurons=1*28*28,
                                                    out_neurons=hidden_neurons,
                                                    threshold=0.9))
        self.add_component('fc2', TravanaeiAndMaida(in_neurons=hidden_neurons,
                                                    out_neurons=num_classes,
                                                    threshold=hidden_neurons*0.025))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        o = self.fc1(o)
        o = self.fc2(o)
        return o

    def reset_variables(self, **kwargs):
        for l in self.layer.values():
            l.reset_variables(**kwargs)


class Cao2015_Tailored(n3ml.network.Network):
    def __init__(self,
                 num_classes: int = 10,
                 in_planes: int = 3,
                 out_planes: int = 64) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.extractor = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_planes, out_planes, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_planes, out_planes, 3, bias=False),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_planes, out_planes, bias=False),
            nn.ReLU(),
            nn.Linear(out_planes, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cao2015_SNN(n3ml.network.Network):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Ho2013(n3ml.network.Network):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TailoredCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, out_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(self.out_channels, self.out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(self.out_channels, self.out_channels, 3, bias=False),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
