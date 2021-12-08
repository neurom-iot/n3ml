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

from n3ml.model import DynamicModel_SpikeNorm_ANN


def validate(val_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)

            loss = criterion(outputs, labels)

            num_corrects += float(torch.argmax(outputs, dim=1).eq(labels).sum())
            total_loss += float(loss)
            total_images += images.size(0)

    val_loss = total_loss / total_images
    val_acc = num_corrects / total_images

    return val_loss, val_acc


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

    model = DynamicModel_SpikeNorm_ANN()
    for m in state_dict['arch']:
        model.add_module(m[0], m[1])

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    loss, acc = validate(val_loader, model, criterion)
    print("In test, loss: {} - acc: {}".format(loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--pretrained', default='pretrained/ann_acc_6666.pt')

    app(parser.parse_args())
