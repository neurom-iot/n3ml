"""
It is the implementation of Spike Norm [Sengupta et al. 2018]. Spike Norm is a threshold
balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion. The implementation involves four python files as follows:
1. 'test_spikenorm_vgg16_train.py' trains a VGG-16 on CIFAR-10 and then saves a trained VGG-16.
2. 'test_spikenorm_vgg16_eval.py' evaluates a trained VGG-16 on CIFAR-10.
3. 'test_spikenorm_svgg16_train.py' trains a spiking VGG-16 using a trained VGG-16 on CIFAR-10
   and then saves the thresholds that found by threshold balancing, Spike Norm algorithm.
4. 'test_spikenorm_svgg16_eval.py' evaluates a trained spiking VGG-16 on CIFAR-10.

Sengupta et al. Going deeper in spiking neural networks: VGG and residual architectures. 2018
"""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from n3ml.model import VGG16, SVGG16


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=opt.batch_size,
        shuffle=False)

    ann = VGG16()
    ann.load_state_dict(torch.load(opt.pretrained)['model'])
    if torch.cuda.is_available():
        ann.cuda()

    snn = SVGG16(ann, batch_size=opt.batch_size)
    snn.eval()
    threshold = torch.load(opt.save)

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

    print("Final validation accuracy: {}".format((num_corrects.float() / total_images).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--num_steps', default=2500, type=int)
    parser.add_argument('--pretrained', default='pretrained/vgg16_acc_9289.pt')
    parser.add_argument('--save', default='pretrained/svgg16_ths_vgg16_acc_9289.pt')

    app(parser.parse_args())

