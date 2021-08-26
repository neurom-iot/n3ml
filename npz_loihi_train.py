import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import n3ml.network


class SCNN(n3ml.network.Network):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, stride=2, bias=False)
        self.fc1 = nn.Linear(in_features=864, out_features=10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def train(train_loader, model, criterion, optimizer):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
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

        if (step+1) % 100 == 0:
            print("step: {} - loss: {} - acc: {}".format(step+1, total_loss/total_images, num_corrects/total_images))


def validate(val_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
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
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    model = SCNN()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    criterion = nn.CrossEntropyLoss()

    best_epoch = 0
    best_acc = 0

    for epoch in range(opt.num_epochs):
        print("epoch: {}".format(epoch+1))
        train(train_loader, model, criterion, optimizer)

        loss, acc = validate(val_loader, model, criterion)
        print("In test, loss: {} - acc: {}".format(loss, acc))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            state_dict = {
                'epoch': best_epoch,
                'best_acc': best_acc,
                'model': model.state_dict()
            }
            torch.save(state_dict, opt.save)

        print()

    print("epoch: {} - best acc: {}".format(best_epoch, best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save', default='pretrained/npz_loihi_train.pt')

    app(parser.parse_args())
