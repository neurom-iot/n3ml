"""
This is the implementation of Input current-based Channel-wise algorithmm [Huynh et al. 2021].
This is a threshold balancing algorithm that finds the proper thresholds of spiking neurons in SNN for ANN-SNN
conversion.

test_current_based_channel_wise_infer.py finds the proper thresholds by using Act_based algorithm and then perform the inference in SNN.

Huynh et al. "....".
"""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from n3ml.model import Diehl2015, SNN_Diehl2015, SNN_Huynh2021
from n3ml.threshold import current_channel_wise


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

    ann = Diehl2015()
    ann.load_state_dict(torch.load(opt.save)['model'])
    ann.eval()
    if torch.cuda.is_available():
        ann.cuda()

    snn = SNN_Huynh2021(batch_size=opt.batch_size)

    saved_state_dict = torch.load(opt.save)
    print(saved_state_dict['epoch'])
    print(saved_state_dict['best_acc'])
    for index, m in enumerate(saved_state_dict['model']):
        snn.state_dict()[m].copy_(saved_state_dict['model'][m])

    snn.eval()
    snn.cuda()
    threshold = current_channel_wise(train_loader=train_loader, model=snn,num_steps=opt.num_steps)
    print(threshold)
    snn.update_threshold(threshold)




    total_images = 0
    num_corrects = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()
            snn.init_neuron_models()
            for t in range  (opt.num_steps):
                outs = snn(images, inference = True)

            # print(labels[0])
            # # print(outs[0])
            # print(outs[0])

            num_corrects += torch.argmax(outs, dim=1).eq(labels).sum(dim=0)
            total_images += images.size(0)

            print("Total images: {} - val. accuracy: {}".format(
                total_images, (num_corrects.float() / total_images).item())
            )

    print("Final validation accuracy: {}".format((num_corrects.float() / total_images).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='test/data')
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--num_steps', default=100, type=int)
    parser.add_argument('--save', default='test/pretrained/diehl2015.pt')
    app(parser.parse_args())
