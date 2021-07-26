import time
import argparse

import torch
import torchvision
import torchvision.transforms

import n3ml.model
import n3ml.encoder


def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: x * 32 * 4)])),
        batch_size=opt.batch_size,
        shuffle=False)

    # Load pretrained weights and thesholds
    state_dict = torch.load(opt.pretrained)['model_state_dict']
    trained_w = state_dict['xe.w']
    trained_th = state_dict['exc.theta']

    model = n3ml.model.DiehlAndCook2015Infer(neurons=opt.neurons)
    model.xe.w.copy_(trained_w)
    model.exc.theta.copy_(trained_th)

    encoder = n3ml.encoder.PoissonEncoder(opt.time_interval)

    total_rates = torch.zeros((opt.num_classes, opt.neurons))
    total_labels = torch.zeros(opt.num_classes)

    start = time.time()

    for step, (image, label) in enumerate(train_loader):
        model.init_param()

        image = image.view(1, 28, 28)

        spiked_image = encoder(image)
        spiked_image = spiked_image.view(opt.time_interval, -1)
        spiked_image = spiked_image.cuda()

        spike_train = []

        for t in range(opt.time_interval):
            model.run({'inp': spiked_image[t]})

            spike_train.append(model.exc.s.clone().detach().cpu())

        spike_train = torch.stack(spike_train)

        total_rates[label] += torch.sum(spike_train, dim=0) / opt.time_interval
        total_labels[label] += 1

        if (step+1) % 1000 == 0:
            end = time.time()
            print("elapsed times: {} - number of images: {}".format(end-start, step+1))

    total_avg_rates = total_rates / total_labels.unsqueeze(dim=1)

    assigned_label = torch.argmax(total_avg_rates, dim=0)

    print(assigned_label)

    torch.save({'assigned_label': assigned_label}, opt.assigned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--pretrained', default='pretrained/stdp_epoch-2_step-60000.pt')
    parser.add_argument('--neurons', default=400, type=int)
    parser.add_argument('--time_interval', default=250, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--assigned', default='assigned/stdp_epoch-2_step-60000.pt')

    app(parser.parse_args())
