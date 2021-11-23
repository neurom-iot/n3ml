import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from n3ml.network import Network
from n3ml.layer import SoftLIF


class Hunsberger2015(Network):
    def __int__(self):
        super(Hunsberger2015, self).__init__()

    def forward(self, x):
        for m in self.named_children():
            x = m[1](x)
        return x


def validate(val_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = criterion(preds, labels)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def train(train_loader, model, criterion, optimizer):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(24),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=opt.batch_size)

    model = Hunsberger2015()

    model.add_module('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False))
    model.add_module('slif1', SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
    model.add_module('apool1', nn.AvgPool2d(kernel_size=2))
    model.add_module('conv2', nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False))
    model.add_module('slif2', SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
    model.add_module('apool2', nn.AvgPool2d(kernel_size=2))
    model.add_module('conv3', nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False))
    model.add_module('slif3', SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
    model.add_module('apool3', nn.AvgPool2d(kernel_size=2))
    model.add_module('flatten4', nn.Flatten())
    model.add_module('drop5', nn.Dropout())
    model.add_module('fc5', nn.Linear(256, 1024, bias=False))
    model.add_module('slif5', SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
    model.add_module('fc6', nn.Linear(1024, opt.num_classes, bias=False))

    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end-start, epoch, train_acc, train_loss))

        val_acc, val_loss = validate(val_loader, model, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()}
            torch.save(state, opt.pretrained)
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, val_loss))

        # lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--pretrained', default='pretrained/softlif_dynamic.pt')

    parser.add_argument('--amplitude', default=0.063, type=float)
    parser.add_argument('--tau_ref', default=0.001, type=float)
    parser.add_argument('--tau_rc', default=0.05, type=float)
    parser.add_argument('--gain', default=0.825, type=float)
    parser.add_argument('--sigma', default=0.02, type=float)

    app(parser.parse_args())
