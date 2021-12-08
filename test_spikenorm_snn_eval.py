"""
It is the implementation of Spike Norm [Sengupta et al. 2018]. Spike Norm is a threshold
balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion.

Sengupta et al. Going deeper in spiking neural networks: VGG and residual architectures. 2018
"""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from n3ml.model import DynamicModel_SpikeNorm_ANN, DynamicModel_SpikeNorm_SNN


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=opt.batch_size,
        shuffle=False)

    state_dict = torch.load(opt.pretrained)

    ann = DynamicModel_SpikeNorm_ANN()
    for m in state_dict['arch']:
        ann.add_module(m[0], m[1])

    threshold = [_.item() for _ in torch.load(opt.save)]
    snn = DynamicModel_SpikeNorm_SNN(ann=ann,
                                     batch_size=opt.batch_size,
                                     fake_x=torch.zeros(size=(1, 3, 32, 32)),
                                     threshold=threshold)
    print(threshold)

    if torch.cuda.is_available():
        snn.cuda()
    snn.eval()

    total_images = 0
    num_corrects = 0

    with torch.no_grad():
        for images, labels in val_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outs = snn(images, num_steps=opt.num_steps)

            print(labels[0])
            print(outs[0])

            num_corrects += torch.argmax(outs, dim=1).eq(labels).sum(dim=0)
            total_images += images.size(0)

            print("Total images: {} - val. accuracy: {}".format(
                total_images, (num_corrects.float() / total_images).item())
            )

    print("Final validation accuracy: {}".format((num_corrects.float() / total_images).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--num_steps', default=2500, type=int)
    parser.add_argument('--pretrained', default='pretrained/ann_acc_8605.pt')
    parser.add_argument('--save', default='pretrained/snn_ths_ann_acc_8605.pt')

    app(parser.parse_args())
