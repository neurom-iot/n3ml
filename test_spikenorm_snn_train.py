"""
It is the implementation of Spike Norm [Sengupta et al. 2018]. Spike Norm is a threshold
balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion.

Sengupta et al. Going deeper in spiking neural networks: VGG and residual architectures. 2018
"""
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from n3ml.model import DynamicModel_SpikeNorm_ANN, DynamicModel_SpikeNorm_SNN
from n3ml.threshold import spikenorm


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
    parser.add_argument('--pretrained', default='pretrained/ann_acc_8605.pt')
    parser.add_argument('--save', default='pretrained/snn_ths_ann_acc_8605.pt')
    parser.add_argument('--scaling_factor', default=1.0, type=float)

    app(parser.parse_args())
