import time
import argparse

import numpy as np

import torch
import torchvision
import torchvision.transforms

import n3ml.model
import n3ml.encoder
import n3ml.visualizer

np.set_printoptions(precision=3, linewidth=np.inf)


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: x * 32 * 4)])),
        batch_size=opt.batch_size,
        shuffle=False)

    state_dict = torch.load(opt.pretrained)['model_state_dict']
    trained_w = state_dict['xe.w']
    trained_th = state_dict['exc.theta']

    model = n3ml.model.DiehlAndCook2015Infer(neurons=opt.neurons)
    model.xe.w.copy_(trained_w)
    model.exc.theta.copy_(trained_th)

    encoder = n3ml.encoder.PoissonEncoder(opt.time_interval)

    assigned_label = torch.load(opt.assigned)['assigned_label']

    num_corrects = 0
    num_images = 0

    start = time.time()

    for step, (image, label) in enumerate(val_loader):
        model.init_param()

        image = image.view(1, 28, 28)

        spiked_image = encoder(image)
        spiked_image = spiked_image.view(opt.time_interval, -1)
        spiked_image = spiked_image.cuda()

        spike_train = []
        total_rates_for_each_class = torch.zeros(opt.num_classes)
        total_labels_for_each_class = torch.zeros(opt.num_classes)

        for t in range(opt.time_interval):
            model.run({'inp': spiked_image[t]})

            spike_train.append(model.exc.s.clone().detach().cpu())

        spike_train = torch.stack(spike_train)

        rates = torch.sum(spike_train, dim=0) / opt.time_interval

        for i in range(rates.size(0)):
            total_rates_for_each_class[assigned_label[i]] += rates[i]
            total_labels_for_each_class[assigned_label[i]] += 1

        avg_rates = total_rates_for_each_class / total_labels_for_each_class

        # print(avg_rates)

        num_corrects += 1 if torch.argmax(avg_rates) == label else 0
        num_images += 1

        if (step+1) % 500 == 0:
            end = time.time()
            print("In {} steps - elapsed time: {} - accuracy: {}".format(step+1, end-start, 1.0*num_corrects/num_images))

    print("total accuracy: {}".format(1.0*num_corrects/num_images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_interval', default=250, type=int)
    parser.add_argument('--pretrained', default='pretrained/stdp_epoch-2_step-60000.pt')
    parser.add_argument('--assigned', default='assigned/stdp_epoch-2_step-60000.pt')
    parser.add_argument('--neurons', default=400, type=int)

    app(parser.parse_args())
