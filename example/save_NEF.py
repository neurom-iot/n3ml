import argparse

import numpy as np

import torch

import n3ml
import n3ml.model


def app(opt):
    print(opt)

    npz = np.load(opt.npz, allow_pickle=True)

    state_dict = {}
    for item in npz:
        state_dict[item] = dict(npz[item].tolist())

    model = n3ml.model.Voelker2015(neurons=state_dict['ens_args']['n_neurons'],
                                   input_size=state_dict['ens_args']['input_dimensions'],
                                   output_size=state_dict['ens_args']['output_dimensions'])
    model.pop.e.copy_(torch.tensor(state_dict['ens_args']['scaled_encoders']))
    model.pop.a.copy_(torch.tensor(state_dict['ens_args']['gain']))
    model.pop.bias.copy_(torch.tensor(state_dict['ens_args']['bias']))
    model.pop.d.copy_(torch.tensor(state_dict['conn_args']['weights']))

    # To check the exception in def n3ml.to_state_dict
    # model.population = {}

    state = n3ml.to_state_dict(dt=0.001, lr=0.01, model=model)

    n3ml.save(state, f=opt.save)

    npz = np.load(opt.save, allow_pickle=True)

    state_dict = {}
    for item in npz:
        state_dict[item] = dict(npz[item].tolist())

    print(state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 실제 실행을 위해서는 경로가 수정되어야 합니다.
    parser.add_argument('--npz', default='data/npz/fpen_args_2975099280.npz')
    parser.add_argument('--save', default='data/npz/n3ml_202108101515.npz')

    app(parser.parse_args())
