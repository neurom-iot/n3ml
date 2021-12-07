"""
test_act_based_train.py:
    - Trains a CNN model and then saves a pretrained model to train SNN.

Diehl et al. "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing.
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from n3ml.model import Diehl2015


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
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    model = Diehl2015()
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            }
            torch.save(state_dict, opt.save)

        print()

    print("epoch: {} - best acc: {}".format(best_epoch, best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='test/data')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--save', default='test/pretrained/diehl2015.pt')

    app(parser.parse_args())
