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
        batch_size=opt.batch_size,
        shuffle=True)

    ann = VGG16()
    ann.load_state_dict(torch.load(opt.pretrained)['model'])
    if torch.cuda.is_available():
        ann.cuda()

    snn = SVGG16(ann, batch_size=opt.batch_size)
    snn.eval()
    threshold = spikenorm(train_loader=train_loader,
                          encoder=lambda x: torch.mul(torch.le(torch.rand_like(x), torch.abs(x)*1.0).float(),
                                                      torch.sign(x)),
                          model=snn, num_steps=opt.num_steps, scaling_factor=opt.scaling_factor)

    torch.save(threshold, opt.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--num_steps', default=500, type=int)
    parser.add_argument('--pretrained', default='pretrained/vgg16_acc_9289.pt')
    parser.add_argument('--save', default='pretrained/svgg16_ths.pt')
    parser.add_argument('--scaling_factor', default=1.0, type=float)

    app(parser.parse_args())
