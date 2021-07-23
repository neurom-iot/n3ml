import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

import n3ml.model
import n3ml.encoder
import n3ml.optimizer

np.set_printoptions(precision=5, linewidth=np.inf)


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

    npz = np.load(opt.npz, allow_pickle=True)

    state_dict = {}

    """
        현재 사용 중인 npz 파일은 다음과 같은 구조로 상태값을 저장한다.
        
        'sim_args'
            'dt': float
        'ens_args'
            'input_dimensions': int
            'output_dimensions': int
            'n_neurons': int
            'bias': ndarray (81,)
            'gain': ndarray (81,)
            'scaled_encoders': ndarray (81, 196)
            'neuron_type': str
        'conn_args'
            'weights': ndarray (10, 81)
            'learning_rate': float
        'recur_args'
            'weights': int (여기서 사용하지 않음)
    """
    for item in npz:
        state_dict[item] = dict(npz[item].tolist())

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(14),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    model = n3ml.model.Voelker2015(neurons=state_dict['ens_args']['n_neurons'],
                                   input_size=state_dict['ens_args']['input_dimensions'],
                                   output_size=state_dict['ens_args']['output_dimensions'])

    model.pop.e.copy_(torch.tensor(state_dict['ens_args']['scaled_encoders']))
    model.pop.a.copy_(torch.tensor(state_dict['ens_args']['gain']))
    model.pop.bias.copy_(torch.tensor(state_dict['ens_args']['bias']))
    model.pop.d.copy_(torch.tensor(state_dict['conn_args']['weights']))

    encoder = n3ml.encoder.Simple(time_interval=opt.time_interval)

    optimizer = n3ml.optimizer.Voelker(model.pop, lr=state_dict['conn_args']['learning_rate'])

    for epoch in range(opt.num_epochs):
        train(loader=train_loader, model=model, encoder=encoder, optimizer=optimizer, opt=opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--npz', default='data/npz/fpen_args_2975099280.npz')
    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_interval', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_classes', default=10, type=int)

    app(parser.parse_args())
