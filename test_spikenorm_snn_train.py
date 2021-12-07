"""
It is the implementation of Spike Norm [Sengupta et al. 2018]. Spike Norm is a threshold
balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion.

Sengupta et al. Going deeper in spiking neural networks: VGG and residual architectures. 2018
"""
from collections import OrderedDict
from typing import List
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from n3ml.layer import SoftIF1d, SoftIF2d
from n3ml.threshold import spikenorm


class DynamicModel_SpikeNorm_ANN(nn.Module):
    def __init__(self) -> None:
        super(DynamicModel_SpikeNorm_ANN, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.named_children():
            x = m[1](x)
        return x


class DynamicModel_SpikeNorm_SNN(nn.Module):
    def __init__(self,
                 ann,
                 batch_size: int,
                 fake_x: torch.Tensor,
                 threshold: List[float]) -> None:
        super(DynamicModel_SpikeNorm_SNN, self).__init__()
        # Count the sizes of ReLUs and then determine its
        # corresponding spiking neuron model - SoftIF1d or
        # SoftIF2d.
        x = fake_x
        if torch.cuda.is_available():
            x = x.cuda()
        sz = OrderedDict()
        for m in ann.named_children():
            x = m[1](x)
            sz[m[0]] = x.size()

        l = 0
        for m in ann.named_children():
            if not isinstance(m[1], nn.ReLU):
                self.add_module(m[0], m[1])
                if isinstance(m[1], nn.Linear) or isinstance(m[1], nn.Conv2d):
                    name = ''.join([m[0], '_', 'if'])
                    ssz = sz[m[0]]
                    if len(ssz) > 2:
                        neuron = SoftIF2d(batch_size=batch_size,
                                          num_channels=ssz[1],
                                          height=ssz[2],
                                          width=ssz[3],
                                          threshold=threshold[l])
                    else:
                        neuron = SoftIF1d(batch_size=batch_size,
                                          num_features=ssz[1],
                                          threshold=threshold[l])
                    l += 1
                    self.add_module(name, neuron)

    def init_neurons(self) -> None:
        for m in self.named_children():
            if isinstance(m[1], SoftIF1d) or isinstance(m[1], SoftIF2d):
                m[1].init_vars()

    def update_threshold(self, threshold: List[float]) -> None:
        l = 0
        for m in self.named_children():
            if isinstance(m[1], SoftIF1d) or isinstance(m[1], SoftIF2d):
                m[1].threshold.fill_(threshold[l])
                l += 1

    def forward(self,
                images: torch.Tensor,
                num_steps: int,
                find_max_input: bool = False,
                find_max_layer: int = 0) -> torch.Tensor:
        self.init_neurons()
        max_mem = 0.0
        o = 0
        for step in range(num_steps):
            x = torch.mul(torch.le(torch.rand_like(images), torch.abs(images)*1.0).float(), torch.sign(images))
            l = 0
            done = False
            for m in self.named_children():
                if isinstance(m[1], SoftIF1d) or isinstance(m[1], SoftIF2d):
                    if find_max_input and find_max_layer == l:
                        if x.max() > max_mem:
                            max_mem = x.max()
                        done = True
                        break
                    l += 1
                x = m[1](x)
            if done:
                continue
            o += x
        if find_max_input:
            return max_mem
        return o


def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=opt.batch_size,
        shuffle=True)

    state_dict = torch.load(opt.pretrained)

    ann = DynamicModel_SpikeNorm_ANN()
    for m in state_dict['arch']:
        ann.add_module(m[0], m[1])

    num_ths = 0
    for m in ann.named_children():
        if isinstance(m[1], nn.ReLU):
            num_ths += 1
    num_ths += 1
    threshold = [1.0] * num_ths
    snn = DynamicModel_SpikeNorm_SNN(ann=ann,
                                     batch_size=opt.batch_size,
                                     fake_x=torch.zeros(size=(1, 3, 32, 32)),
                                     threshold=threshold)
    if torch.cuda.is_available():
        snn.cuda()
    snn.eval()

    threshold = spikenorm(train_loader=train_loader,
                          encoder=lambda x: torch.mul(torch.le(torch.rand_like(x), torch.abs(x)*1.0).float(),
                                                      torch.sign(x)),
                          model=snn, num_steps=opt.num_steps, scaling_factor=opt.scaling_factor)

    print("The found thresholds by Spike Norm are\n{}".format(threshold))

    torch.save(threshold, opt.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_steps', default=500, type=int)
    parser.add_argument('--pretrained', default='pretrained/ann_acc_8676.pt')
    parser.add_argument('--save', default='pretrained/snn_ths_ann_acc_8676.pt')
    parser.add_argument('--scaling_factor', default=1.0, type=float)

    app(parser.parse_args())
