"""
    Bohte2002 example.

    Now, spikeprop algorithm can be applied to only feed-forward neural network.
    It means that when we construct neural network only consider a sequential structure.
    We achieve this using nn.Sequential.
"""
import argparse

import numpy as np

import matplotlib.pyplot as plt

import torch

import n3ml.model
import n3ml.data
import n3ml.encoder
import n3ml.optimizer


class Plot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax2 = self.ax.twinx()
        plt.title('SpikeProp')

    def update(self, y1, y2):
        x = torch.arange(y1.shape[0]) * 30

        ax1 = self.ax
        ax2 = self.ax2

        ax1.plot(x, y1, 'g')
        ax2.plot(x, y2, 'b')

        ax1.set_xlabel('number of images')
        ax1.set_ylabel('accuracy', color='g')
        ax2.set_ylabel('loss', color='b')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class LabelEncoder:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def run(self, label):
        o = torch.zeros(self.num_classes)
        o.fill_(13)  # 15 13
        o[label].fill_(8)  # 5 7
        return o


def rmse(pred, target):
    # print("pred: {} - target: {}".format(pred, target))
    return torch.sum((pred[pred >= 0]-target[pred >= 0])**2)/2


def do_correct(o, y):
    if torch.argmin(o) == (o.size(0)-torch.argmin(torch.flip(o, [0]))-1):
        return torch.argmin(o) == torch.argmin(y)
    return torch.tensor(False)


def validate(data, model, data_encoder, label_encoder, loss, opt):
    total_data = 0
    corrects = 0
    total_loss = 0

    for i in range(data['test.data'].size(0)):
        model.initialize(delay=False)

        input = data['test.data'][i]
        label = data['test.target'][i]

        spiked_input = data_encoder.run(input)
        spiked_input = torch.cat((spiked_input.view(-1), torch.zeros(2)))
        spiked_label = label_encoder.run(label)

        for t in range(opt.num_steps):
            model(torch.tensor(t).float(), spiked_input)
        o = model.fc2.s

        total_data += 1
        corrects += do_correct(o, spiked_label)
        total_loss += loss(o, spiked_label)

    avg_acc = corrects.float() / total_data
    avg_loss = total_loss / total_data

    return avg_loss, avg_acc


def train(data, model, data_encoder, label_encoder, optimizer, loss, epoch, meter, acc_buffer, loss_buffer, plotter, opt):
    for i in range(data['train.data'].size(0)):
        model.initialize(delay=False)

        input = data['train.data'][i]
        label = data['train.target'][i]

        spiked_input = data_encoder.run(input)
        spiked_input = torch.cat((spiked_input.view(-1), torch.zeros(2)))
        spiked_label = label_encoder.run(label)

        for t in range(opt.num_steps):
            model(torch.tensor(t).float(), spiked_input)
        o = model.fc2.s

        # print(model.fc1.s)
        # print(model.fc2.s)
        # print("pred: {} - target: {}".format(o, spiked_label))
        # l = loss(o, spiked_label)
        # print("loss: {}".format(l))

        optimizer.step(model, spiked_input, spiked_label, epoch)

        meter['num_images'] += 1
        meter['num_corrects'] += do_correct(o, spiked_label)
        meter['total_losses'] += loss(o, spiked_label)

        if (i+1) % 30 == 0:
            print("label: {} - target: {} - pred: {} - result: {}".format(label, spiked_label, o, do_correct(o, spiked_label)))

            acc_buffer.append(1.0*meter['num_corrects']/meter['num_images'])
            loss_buffer.append(meter['total_losses']/meter['num_images'])

            plotter.update(y1=np.array(acc_buffer), y2=np.array(loss_buffer))


def app(opt):
    np.set_printoptions(threshold=np.inf)

    print(opt)

    data_loader = n3ml.data.IRISDataLoader(ratio=0.8)
    data = data_loader.run()
    summary = data_loader.summarize()

    data_encoder = n3ml.encoder.Population(neurons=12,
                                           minimum=summary['min'],
                                           maximum=summary['max'],
                                           max_firing_time=opt.max_firing_time,
                                           not_to_fire=opt.not_to_fire,
                                           dt=opt.dt)
    label_encoder = LabelEncoder(opt.num_classes)

    model = n3ml.model.Bohte2002()
    model.initialize()

    optimizer = n3ml.optimizer.Bohte()

    # for plot
    plotter = Plot()

    meter = {
        'total_losses': 0.0,
        'num_corrects': 0,
        'num_images': 0
    }

    acc_buffer = []
    loss_buffer = []

    for epoch in range(opt.num_epochs):
        train(data, model, data_encoder, label_encoder, optimizer, rmse, epoch, meter, acc_buffer, loss_buffer, plotter, opt)
        # print("epoch: {} - tr. loss: {} - tr. acc: {}".format(epoch, loss, acc))

        loss, acc = validate(data, model, data_encoder, label_encoder, rmse, opt)
        print("epoch: {} - val. loss: {} - val. acc: {}".format(epoch, loss, acc))

        data = data_loader.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--dt', default=1, type=int)
    parser.add_argument('--num_steps', default=40, type=int)
    parser.add_argument('--max_firing_time', default=30, type=int)
    parser.add_argument('--not_to_fire', default=28, type=int)

    app(parser.parse_args())
