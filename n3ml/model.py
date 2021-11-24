import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.distributions.uniform

from n3ml.layer import IF1d, IF2d, Conv2d, AvgPool2d, Linear, Bohte, TravanaeiAndMaida

from n3ml.network import Network
import n3ml.layer
import n3ml.population
import n3ml.connection
import n3ml.learning
from n3ml.layer import SoftLIF, LIF1d, LIF2d, SoftIF1d, SoftIF2d


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
                    self.add_module('{}.LIF2d'.format(m[0]), LIF2d(batch_size=shapes[m[0]][0],
                                                                   plains=shapes[m[0]][1],
                                                                   height=shapes[m[0]][2],
                                                                   width=shapes[m[0]][3]))
                elif isinstance(m[1], nn.Linear):
                    self.add_module('{}.LIF1d'.format(m[0]), LIF1d(batch_size=shapes[m[0]][0],
                                                                   neurons=shapes[m[0]][1]))

    def reset_variables(self, batch_size: int) -> None:
        for m in self.named_modules():
            if isinstance(m[1], LIF1d) or isinstance(m[1], LIF2d):
                m[1].reset_variables(batch_size=batch_size)

    def update_thresholds(self, thresholds: List[float]) -> None:
        pass

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


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.conv1_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_relu = nn.ReLU(inplace=True)
        self.conv2_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_relu = nn.ReLU(inplace=True)
        self.conv3_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_relu = nn.ReLU(inplace=True)
        self.conv4_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_relu = nn.ReLU(inplace=True)
        self.conv5_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_relu = nn.ReLU(inplace=True)
        self.conv6_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7_relu = nn.ReLU(inplace=True)
        self.conv7_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8_relu = nn.ReLU(inplace=True)
        self.conv8_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9_relu = nn.ReLU(inplace=True)
        self.conv9_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10_relu = nn.ReLU(inplace=True)
        self.conv10_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv11_relu = nn.ReLU(inplace=True)
        self.conv11_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12_relu = nn.ReLU(inplace=True)
        self.conv12_drop = nn.Dropout(p=0.2, inplace=False)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv13_relu = nn.ReLU(inplace=True)
        self.conv13_drop = nn.Dropout(p=0.2, inplace=False)
        self.fc14 = nn.Linear(in_features=2048, out_features=4096, bias=False)
        self.fc14_relu = nn.ReLU(inplace=True)
        self.fc14_drop = nn.Dropout(p=0.5, inplace=False)
        self.fc15 = nn.Linear(in_features=4096, out_features=4096, bias=False)
        self.fc15_relu = nn.ReLU(inplace=True)
        self.fc15_drop = nn.Dropout(p=0.5, inplace=False)
        self.fc16 = nn.Linear(in_features=4096, out_features=10, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        #############################
        #   @author: Nitin Rathi    #
        #############################
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_drop(x)
        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_avgpool(x)
        x = self.conv3(x)
        x = self.conv3_relu(x)
        x = self.conv3_drop(x)
        x = self.conv4(x)
        x = self.conv4_relu(x)
        x = self.conv4_avgpool(x)
        x = self.conv5(x)
        x = self.conv5_relu(x)
        x = self.conv5_drop(x)
        x = self.conv6(x)
        x = self.conv6_relu(x)
        x = self.conv6_drop(x)
        x = self.conv7(x)
        x = self.conv7_relu(x)
        x = self.conv7_avgpool(x)
        x = self.conv8(x)
        x = self.conv8_relu(x)
        x = self.conv8_drop(x)
        x = self.conv9(x)
        x = self.conv9_relu(x)
        x = self.conv9_drop(x)
        x = self.conv10(x)
        x = self.conv10_relu(x)
        x = self.conv10_avgpool(x)
        x = self.conv11(x)
        x = self.conv11_relu(x)
        x = self.conv11_drop(x)
        x = self.conv12(x)
        x = self.conv12_relu(x)
        x = self.conv12_drop(x)
        x = self.conv13(x)
        x = self.conv13_relu(x)
        x = self.conv13_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc14(x)
        x = self.fc14_relu(x)
        x = self.fc14_drop(x)
        x = self.fc15(x)
        x = self.fc15_relu(x)
        x = self.fc15_drop(x)
        x = self.fc16(x)
        return x


class SVGG16(nn.Module):
    def __init__(self, vgg16, batch_size: int, threshold: List[float] = None) -> None:
        super(SVGG16, self).__init__()
        self.batch_size = batch_size
        if not threshold:
            threshold = [1.0] * 16
        self.threshold = threshold
        self.conv1 = vgg16.conv1
        self.conv1_if = SoftIF2d(batch_size=batch_size, num_channels=64, height=32, width=32, threshold=threshold[0])
        self.conv1_drop = vgg16.conv1_drop
        self.conv2 = vgg16.conv2
        self.conv2_if = SoftIF2d(batch_size=batch_size, num_channels=64, height=32, width=32, threshold=threshold[1])
        self.conv2_avgpool = vgg16.conv2_avgpool
        self.conv3 = vgg16.conv3
        self.conv3_if = SoftIF2d(batch_size=batch_size, num_channels=128, height=16, width=16, threshold=threshold[2])
        self.conv3_drop = vgg16.conv3_drop
        self.conv4 = vgg16.conv4
        self.conv4_if = SoftIF2d(batch_size=batch_size, num_channels=128, height=16, width=16, threshold=threshold[3])
        self.conv4_avgpool = vgg16.conv4_avgpool
        self.conv5 = vgg16.conv5
        self.conv5_if = SoftIF2d(batch_size=batch_size, num_channels=256, height=8, width=8, threshold=threshold[4])
        self.conv5_drop = vgg16.conv5_drop
        self.conv6 = vgg16.conv6
        self.conv6_if = SoftIF2d(batch_size=batch_size, num_channels=256, height=8, width=8, threshold=threshold[5])
        self.conv6_drop = vgg16.conv6_drop
        self.conv7 = vgg16.conv7
        self.conv7_if = SoftIF2d(batch_size=batch_size, num_channels=256, height=8, width=8, threshold=threshold[6])
        self.conv7_avgpool = vgg16.conv7_avgpool
        self.conv8 = vgg16.conv8
        self.conv8_if = SoftIF2d(batch_size=batch_size, num_channels=512, height=4, width=4, threshold=threshold[7])
        self.conv8_drop = vgg16.conv8_drop
        self.conv9 = vgg16.conv9
        self.conv9_if = SoftIF2d(batch_size=batch_size, num_channels=512, height=4, width=4, threshold=threshold[8])
        self.conv9_drop = vgg16.conv9_drop
        self.conv10 = vgg16.conv10
        self.conv10_if = SoftIF2d(batch_size=batch_size, num_channels=512, height=4, width=4, threshold=threshold[9])
        self.conv10_avgpool = vgg16.conv10_avgpool
        self.conv11 = vgg16.conv11
        self.conv11_if = SoftIF2d(batch_size=batch_size, num_channels=512, height=2, width=2, threshold=threshold[10])
        self.conv11_drop = vgg16.conv11_drop
        self.conv12 = vgg16.conv12
        self.conv12_if = SoftIF2d(batch_size=batch_size, num_channels=512, height=2, width=2, threshold=threshold[11])
        self.conv12_drop = vgg16.conv12_drop
        self.conv13 = vgg16.conv13
        self.conv13_if = SoftIF2d(batch_size=batch_size, num_channels=512, height=2, width=2, threshold=threshold[12])
        self.conv13_drop = vgg16.conv13_drop
        self.flat = nn.Flatten()
        self.fc14 = vgg16.fc14
        self.fc14_if = SoftIF1d(batch_size=batch_size, num_features=4096, threshold=threshold[13])
        self.fc14_drop = vgg16.fc14_drop
        self.fc15 = vgg16.fc15
        self.fc15_if = SoftIF1d(batch_size=batch_size, num_features=4096, threshold=threshold[14])
        self.fc15_drop = vgg16.fc15_drop
        self.fc16 = vgg16.fc16
        self.fc16_if = SoftIF1d(batch_size=batch_size, num_features=10, threshold=threshold[15])

    def update_threshold(self, threshold: List[float]) -> None:
        self.conv1_if.threshold = threshold[0]
        self.conv2_if.threshold = threshold[1]
        self.conv3_if.threshold = threshold[2]
        self.conv4_if.threshold = threshold[3]
        self.conv5_if.threshold = threshold[4]
        self.conv6_if.threshold = threshold[5]
        self.conv7_if.threshold = threshold[6]
        self.conv8_if.threshold = threshold[7]
        self.conv9_if.threshold = threshold[8]
        self.conv10_if.threshold = threshold[9]
        self.conv11_if.threshold = threshold[10]
        self.conv12_if.threshold = threshold[11]
        self.conv13_if.threshold = threshold[12]
        self.fc14_if.threshold = threshold[13]
        self.fc15_if.threshold = threshold[14]
        self.fc16_if.threshold = threshold[15]

    def forward(self, images: torch.Tensor,
                num_steps: int,
                find_max_inp: bool = False,
                find_max_layer: int = 0) -> Union[torch.Tensor, float]:

        self.init_neuron_models()
        max_mem = 0.0
        o = 0

        for step in range(num_steps):
            x = torch.mul(torch.le(torch.rand_like(images), torch.abs(images)*1.0).float(), torch.sign(images))

            x = self.conv1(x)
            if find_max_inp and find_max_layer == 0:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv1_if(x)
            x = self.conv1_drop(x)
            x = self.conv2(x)
            if find_max_inp and find_max_layer == 1:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv2_if(x)
            x = self.conv2_avgpool(x)
            x = self.conv3(x)
            if find_max_inp and find_max_layer == 2:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv3_if(x)
            x = self.conv3_drop(x)
            x = self.conv4(x)
            if find_max_inp and find_max_layer == 3:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv4_if(x)
            x = self.conv4_avgpool(x)
            x = self.conv5(x)
            if find_max_inp and find_max_layer == 4:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv5_if(x)
            x = self.conv5_drop(x)
            x = self.conv6(x)
            if find_max_inp and find_max_layer == 5:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv6_if(x)
            x = self.conv6_drop(x)
            x = self.conv7(x)
            if find_max_inp and find_max_layer == 6:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv7_if(x)
            x = self.conv7_avgpool(x)
            x = self.conv8(x)
            if find_max_inp and find_max_layer == 7:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv8_if(x)
            x = self.conv8_drop(x)
            x = self.conv9(x)
            if find_max_inp and find_max_layer == 8:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv9_if(x)
            x = self.conv9_drop(x)
            x = self.conv10(x)
            if find_max_inp and find_max_layer == 9:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv10_if(x)
            x = self.conv10_avgpool(x)
            x = self.conv11(x)
            if find_max_inp and find_max_layer == 10:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv11_if(x)
            x = self.conv11_drop(x)
            x = self.conv12(x)
            if find_max_inp and find_max_layer == 11:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv12_if(x)
            x = self.conv12_drop(x)
            x = self.conv13(x)
            if find_max_inp and find_max_layer == 12:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.conv13_if(x)
            x = self.conv13_drop(x)
            x = self.flat(x)
            x = self.fc14(x)
            if find_max_inp and find_max_layer == 13:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.fc14_if(x)
            x = self.fc14_drop(x)
            x = self.fc15(x)
            if find_max_inp and find_max_layer == 14:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.fc15_if(x)
            x = self.fc15_drop(x)
            x = self.fc16(x)
            if find_max_inp and find_max_layer == 15:
                if x.max() > max_mem:
                    max_mem = x.max()
                continue
            x = self.fc16_if(x)

            o += x

        if find_max_inp:
            return max_mem
        return o

    def init_neuron_models(self):
        for m in self.named_children():
            if isinstance(m[1], SoftIF1d) or isinstance(m[1], SoftIF2d):
                m[1].init_vars()


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
