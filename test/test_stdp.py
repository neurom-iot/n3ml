import time
import argparse

import torch
import torchvision
from torchvision.transforms import transforms

from n3ml.model import DiehlAndCook2015
from n3ml.visualizer import plot
from n3ml.encoder import PoissonEncoder


def app(opt):
    print(opt)

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(), transforms.Lambda(lambda x: x * 32)])),
        batch_size=opt.batch_size,
        shuffle=True)

    # Load fashion-MNIST dataset
    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.FashionMNIST(
    #         opt.data,
    #         train=True,
    #         download=True,
    #         transform=torchvision.transforms.Compose([
    #             transforms.ToTensor(), transforms.Lambda(lambda x: x * 32)])),
    #     batch_size=opt.batch_size,
    #     shuffle=True)

    # Define an encoder to generate spike train for an image
    encoder = PoissonEncoder(opt.time_interval)

    # Define a model
    model = DiehlAndCook2015(neurons=100).cuda()

    fig = None
    mat = None

    i = 0

    import numpy as np
    np.set_printoptions(precision=3, linewidth=np.inf)

    # Conduct training phase
    for epoch in range(opt.num_epochs):
        start = time.time()
        for images, labels in train_loader:
            if i % 1000 == 0:
                print("{}-th images are used to train".format(i + 1))

            # Initialize a model
            model.init_param()

            # print(images.view(28, 28).detach().numpy())

            # Encode images into spiked_images
            images = images.view(1, 28, 28)

            spiked_images = encoder(images)
            spiked_images = spiked_images.view(opt.time_interval, -1)
            spiked_images = spiked_images.cuda()

            # Train a model
            for t in range(opt.time_interval):
                # print(spiked_images[t].detach().cpu().numpy().reshape(28, 28))

                model.run({'inp': spiked_images[t]})

                # print(model.inp.s.cpu().numpy().reshape(28, 28))
                # print(model.exc.v)
                # print(model.exc.s.cpu().numpy())

                # Update weights using learning rule
                model.update()

            # Normalize weights
            model.normalize()

            w = model.xe.w.detach().cpu().numpy()
            fig, mat = plot(fig, mat, w)

            i += 1
        end = time.time()
        print("For one epoch, elapsed times: {}".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_step', default=1, type=int)         # 1ms
    parser.add_argument('--time_interval', default=250, type=int)   # 250ms

    parser.add_argument('--num_epochs', default=5, type=int)

    app(parser.parse_args())
