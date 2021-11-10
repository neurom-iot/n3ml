import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from n3ml.network import Network
from n3ml.layer import Wu1d, Wu2d


class Wu2018(Network):
    def __int__(self, batch_size: int) -> None:
        super(Wu2018, self).__init__()
        self.batch_size = batch_size

    def forward(self, images: torch.Tensor, num_steps: int) -> torch.Tensor:
        v = {}
        s = {}
        for m in self.named_children():
            if isinstance(m[1], Wu1d):
                v[m[0]] = torch.zeros(m[1].batch_size, m[1].neurons, device=images.device)
                s[m[0]] = torch.zeros(m[1].batch_size, m[1].neurons, device=images.device)
            elif isinstance(m[1], Wu2d):
                v[m[0]] = torch.zeros(m[1].batch_size, m[1].planes, m[1].height, m[1].width, device=images.device)
                s[m[0]] = torch.zeros(m[1].batch_size, m[1].planes, m[1].height, m[1].width, device=images.device)
        o = []
        for t in range(num_steps):
            x = (images > torch.rand(images.size(), device=images.device)).float()
            for m in self.named_children():
                if isinstance(m[1], Wu1d) or isinstance(m[1], Wu2d):
                    x = m[1](x, v[m[0]], s[m[0]])
                else:
                    x = m[1](x)
            o.append(x.clone())
        o = torch.stack(o).sum(dim=0) / num_steps
        return o


def validate(val_loader, model, criterion, opt):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            model.init()
            preds = model(images, opt.num_steps)
            labels_ = torch.zeros(torch.numel(labels), 10, device=labels.device)
            labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

            loss = criterion(preds, labels_)

            num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
            total_loss += loss.cpu().detach().numpy() * images.size(0)
            total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def train(train_loader, model, criterion, optimizer, opt):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        model.init()
        preds = model(images, opt.num_steps)

        labels_ = torch.zeros(torch.numel(labels), 10, device=labels.device)
        labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

        o = preds.detach().cpu().numpy()[0]

        print("label: {} - output neuron's voltages: {}".format(labels.detach().cpu().numpy()[0], o))

        loss = criterion(preds, labels_)

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
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size)

    model = Wu2018()

    model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1))
    model.add_module('lif1', Wu2d(opt.batch_size, 32, 28, 28))
    model.add_module('apool1', nn.AvgPool2d(2))
    model.add_module('conv2', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
    model.add_module('lif2', Wu2d(opt.batch_size, 32, 14, 14))
    model.add_module('apool2', nn.AvgPool2d(2))
    model.add_module('flatten3', nn.Flatten())
    model.add_module('fc4', nn.Linear(7 * 7 * 32, 128))
    model.add_module('lif4', Wu1d(opt.batch_size, 128))
    model.add_module('fc5', nn.Linear(128, 10))
    model.add_module('lif5', Wu1d(opt.batch_size, 10))

    model.cuda()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, opt)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end - start, epoch, train_acc,
                                                                                 train_loss))

        val_acc, val_loss = validate(val_loader, model, criterion, opt)

        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()}

            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, val_loss))

        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_steps', default=15, type=int)
    parser.add_argument('--lr', default=1e-03, type=float)

    app(parser.parse_args())
