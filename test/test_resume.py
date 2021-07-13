import argparse

import torch
import torchvision
import torchvision.transforms

import n3ml.model


def train(loader, model, encoder, optimizer):
    for image, label in loader:
        pass


def app(opt):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    model = n3ml.model.ReSuMe2005()

    encoder = None

    optimizer = None

    for epoch in range(opt.num_epochs):
        train(loader=train_loader, model=model, encoder=encoder, optimizer=optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)

    app(parser.parse_args())
