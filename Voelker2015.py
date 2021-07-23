import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

import n3ml.model
import n3ml.encoder
import n3ml.optimizer

np.set_printoptions(precision=3, linewidth=np.inf)


def train(loader, model, encoder, optimizer, opt):
    for image, label in loader:
        # model.init_vars()  # 뭐지? 이거 왜 했지?

        image = image.squeeze(dim=0)
        image = image.view(-1)

        """
            label one-hot encoding 필요
        """
        label = F.one_hot(label, num_classes=opt.num_classes).squeeze(dim=0)

        # spiked_image = encoder(image)
        # spiked_image = spiked_image.view(spiked_image.size(0), -1)

        for t in range(opt.time_interval):
            # print(spiked_image[t].view(28, 28).numpy())  # verified

            model.run({'pop': image})
            o = model.pop.s

            print(label.numpy())
            print(o.numpy())
            # print()

            optimizer.step(o, label)


def app(opt):
    print(opt)

    # Load MNIST
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    # Define model
    model = n3ml.model.Voelker2015(neurons=100,
                                   input_size=784,
                                   output_size=10)

    encoder = n3ml.encoder.Simple(time_interval=opt.time_interval)

    optimizer = n3ml.optimizer.Voelker(model.pop, lr=opt.lr)

    for epoch in range(opt.num_epochs):
        train(loader=train_loader, model=model, encoder=encoder, optimizer=optimizer, opt=opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--time_interval', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    app(parser.parse_args())
