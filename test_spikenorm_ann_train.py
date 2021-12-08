"""
It is the implementation of Spike Norm [Sengupta et al. 2018]. Spike Norm is a threshold
balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion.

Sengupta et al. Going deeper in spiking neural networks: VGG and residual architectures. 2018
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from n3ml.model import DynamicModel_SpikeNorm_ANN


def build_model(model):
    model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
    model.add_module('conv1_relu', nn.ReLU())
    model.add_module('conv1_pool', nn.MaxPool2d(kernel_size=2))
    model.add_module('conv2', nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False))
    model.add_module('conv2_relu', nn.ReLU())
    model.add_module('conv2_pool', nn.MaxPool2d(kernel_size=2))
    model.add_module('conv3', nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1, bias=False))
    model.add_module('conv3_relu', nn.ReLU())
    model.add_module('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, bias=False))
    model.add_module('conv4_relu', nn.ReLU())
    model.add_module('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False))
    model.add_module('conv5_relu', nn.ReLU())
    model.add_module('conv5_pool', nn.MaxPool2d(kernel_size=2))
    model.add_module('flat', nn.Flatten())
    model.add_module('fc6_drop', nn.Dropout())
    model.add_module('fc6', nn.Linear(in_features=256 * 2 * 2, out_features=4096, bias=False))
    model.add_module('fc6_relu', nn.ReLU())
    model.add_module('fc7_drop', nn.Dropout())
    model.add_module('fc7', nn.Linear(in_features=4096, out_features=4096, bias=False))
    model.add_module('fc7_relu', nn.ReLU())
    model.add_module('fc8', nn.Linear(in_features=4096, out_features=10, bias=False))
    return model


def train(train_loader, model, criterion, optimizer):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += float(torch.argmax(outputs, dim=1).eq(labels).sum())
        total_loss += float(loss)
        total_images += images.size(0)

        if (step + 1) % 100 == 0:
            print("step: {} - loss: {} - acc: {}".format(step + 1, total_loss / total_images, num_corrects / total_images))


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

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=opt.batch_size,
        shuffle=False)

    model = DynamicModel_SpikeNorm_ANN()
    model = build_model(model)
    if torch.cuda.is_available():
        model.cuda()

    arch = []
    for m in model.named_children():
        arch.append(m)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[180, 240, 270], gamma=0.1)

    best_epoch = -1
    best_acc = 0

    for epoch in range(opt.num_epochs):
        print("epoch: {}".format(epoch))
        train(train_loader, model, criterion, optimizer)

        loss, acc = validate(val_loader, model, criterion)
        print("In test, loss: {} - acc: {}".format(loss, acc))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            state_dict = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': model.state_dict(),
                'arch': arch
            }
            torch.save(state_dict, opt.save)

        scheduler.step()

        print()

    print("epoch: {} - best acc: {}".format(best_epoch, best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--save', default='pretrained/ann.pt')

    app(parser.parse_args())
