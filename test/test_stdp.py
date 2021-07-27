import time
import argparse

import numpy as np

import torch
import torchvision
from torchvision.transforms import transforms

from n3ml.model import DiehlAndCook2015
from n3ml.visualizer import plot
from n3ml.encoder import PoissonEncoder

np.set_printoptions(precision=3, linewidth=np.inf)


def app(opt):
    print(opt)

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                # transforms.ToTensor(), transforms.Lambda(lambda x: x * 32)])),
                transforms.ToTensor(), transforms.Lambda(lambda x: x * 32 * 4)])),
        batch_size=opt.batch_size,
        shuffle=True)

    # Define an encoder to generate spike train for an image
    encoder = PoissonEncoder(opt.time_interval)

    # Define a model
    model = DiehlAndCook2015(neurons=opt.neurons).cuda()

    fig = None
    mat = None

    # Conduct training phase
    for epoch in range(opt.num_epochs):
        start = time.time()
        for step, (images, labels) in enumerate(train_loader):
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

            if (step+1) % 500 == 0:  # 500 images에 약 250 seconds
                # check training time
                end = time.time()
                print("elpased time: {}".format(end-start))
                print("{} images are used to train".format(step+1))

                # save model
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict()
                }, 'pretrained/stdp_epoch-{}_step-{}.pt'.format(epoch, step+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_step', default=1, type=int)         # 1ms
    parser.add_argument('--time_interval', default=250, type=int)   # 250ms
    parser.add_argument('--neurons', default=400, type=int)

    parser.add_argument('--num_epochs', default=3, type=int)

    app(parser.parse_args())
