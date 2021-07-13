import numpy as np
import time
import argparse

import torch
import torchvision
from torchvision.transforms import transforms

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import n3ml.model


class Plot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax2 = self.ax.twinx()
        plt.title('STBP')

    def update(self, y1, y2):
        x = torch.arange(y1.shape[0]) * 64

        ax1 = self.ax
        ax2 = self.ax2

        ax1.plot(x, y1, 'g')
        ax2.plot(x, y2, 'b')

        ax1.set_xlabel('number of images')
        ax1.set_ylabel('accuracy', color='g')
        ax2.set_ylabel('loss', color='b')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def validate(val_loader, model, criterion):

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        labels_ = torch.zeros(torch.numel(labels), 10).cuda()
        labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

        loss = criterion(preds, labels_)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def train(train_loader, model, criterion, optimizer):
    plotter = Plot()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    list_loss = []
    list_acc = []

    for step, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        labels_ = torch.zeros(torch.numel(labels), 10).cuda()
        labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

        # print("label: {} - prediction: {}".format(labels.detach().cpu().numpy()[0], preds.detach().cpu().numpy()[0]))

        o = preds.detach().cpu().numpy()[0]

        # print("label: {} - prediction:\n".format(labels.detach().cpu().numpy()[0]))
        # pp.print("neuron's index\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9")
        # print("neuron's voltages\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9]))

        print("label: {} - output neuron's voltages: {}".format(labels.detach().cpu().numpy()[0], o))

        loss = criterion(preds, labels_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss   += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

        if total_images > 0:  #  and total_images % 30 == 0
            list_loss.append(total_loss / total_images)
            list_acc.append(float(num_corrects) / total_images)
            plotter.update(y1=np.array(list_acc), y2=np.array(list_loss))


    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def app(opt):
    print(opt)

    # Load MNIST / FashionMNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            download = True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(), transforms.Lambda(lambda x: x * 32)])),
        drop_last = True,
        batch_size=opt.batch_size,
        shuffle=True)

    # Load MNIST/ FashionMNIST dataset
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(), transforms.Lambda(lambda x: x * 32)])),
        drop_last=True,
        batch_size=opt.batch_size,
        shuffle=True)


    model = n3ml.model.Wu2018(batch_size=opt.batch_size, time_interval=opt.time_interval).cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])
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

            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, val_loss))

        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',          default='data')
    parser.add_argument('--num_classes',   default=10,    type=int)
    parser.add_argument('--num_epochs',    default=120,   type=int)
    parser.add_argument('--batch_size',    default=64,    type=int)
    parser.add_argument('--num_workers',   default=8,     type=int)
    parser.add_argument('--time_interval', default=5,     type=int)
    parser.add_argument('--lr',            default=1e-03, type=float)

    app(parser.parse_args())
