import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from n3ml.model import TravanaeiAndMaida2017
from n3ml.encoder import Simple


def accuracy(r: torch.Tensor, label: int) -> torch.Tensor:
    """
    :param r: (time interval, # classes) the spike trains of output neurons in T ms
    :param label:
    :return:
    """
    return (torch.argmax(torch.sum(r, dim=0)) == label).float()


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    model = TravanaeiAndMaida2017(num_classes=opt.num_classes, hidden_neurons=opt.hidden_neurons)
    model.load_state_dict(torch.load(opt.save))

    encoder = Simple(time_interval=opt.time_interval)

    num_images = 0
    num_corrects = 0

    for image, label in val_loader:
        image = image.squeeze(dim=0)
        label = label.squeeze()

        spiked_image = encoder(image)
        spiked_image = spiked_image.view(spiked_image.size(0), -1)

        loss_buffer = []

        for t in range(opt.time_interval):
            model(spiked_image[t])

            loss_buffer.append(model.fc2.o.clone())

        model.reset_variables(w=False)

        num_images += 1
        num_corrects += accuracy(r=torch.stack(loss_buffer), label=label)

    val_acc = float(num_corrects) / num_images
    print("In test, accuracy: {}".format(val_acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--save', default='pretrained/bpstdp.pt')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--hidden_neurons', default=500, type=int)


    app(parser.parse_args())
