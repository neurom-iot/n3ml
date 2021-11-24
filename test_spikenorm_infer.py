"""
It is the implementation of Spike Norm [Sengupta et al. 2018]. Spike Norm is a threshold
balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion.

test_spikenorm_train.py trains a VGG-16 and then saves a pretrained VGG-16 to train SNN.
test_spikenorm_infer.py finds the proper thresholds by using Spike Norm and then train a
spiking VGG-16 using ANN-SNN conversion.

Sengupta et al. Going deeper in spiking neural networks: VGG and residual architectures. 2018
"""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from n3ml.model import VGG16, SVGG16
from n3ml.threshold import spikenorm


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
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=opt.batch_size,
        shuffle=False)

    ann = VGG16()
    ann.load_state_dict(torch.load(opt.save)['model'])
    if torch.cuda.is_available():
        ann.cuda()

    snn = SVGG16(ann, batch_size=opt.batch_size)
    snn.eval()
    threshold = spikenorm(train_loader=train_loader,
                          encoder=lambda x: torch.mul(torch.le(torch.rand_like(x), torch.abs(x)*1.0).float(),
                                                      torch.sign(x)),
                          model=snn,
                          num_steps=opt.num_steps)

    snn.update_threshold(threshold)

    total_images = 0
    num_corrects = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            outs = snn(images, num_steps=opt.num_steps)

            print(labels[0])
            print(outs[0])

            num_corrects += torch.argmax(outs, dim=1).eq(labels).sum(dim=0)
            total_images += images.size(0)

            print("Total images: {} - val. accuracy: {}".format(
                total_images, (num_corrects.float() / total_images).item())
            )

    print("Final validation accuracy: {}".format((num_corrects / total_images).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--num_steps', default=500, type=int)
    parser.add_argument('--save', default='pretrained/vgg16_acc_9289.pt')
    parser.add_argument('--scaling', default=1.0, type=float)  # TODO: Implement

    app(parser.parse_args())
