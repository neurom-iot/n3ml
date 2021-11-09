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
        state_dict[item] = npz[item].tolist()

    model = n3ml.model.Voelker2015(neurons=state_dict['ens_args']['n_neurons'],
                                   input_size=state_dict['ens_args']['input_dimensions'],
                                   output_size=state_dict['ens_args']['output_dimensions'],
                                   neuron_type=state_dict['ens_args']['neuron_type'],
                                   dt=state_dict['sim_args']['dt'] * 1000)
    model.pop.e.copy_(torch.tensor(state_dict['ens_args']['scaled_encoders']))
    model.pop.a.copy_(torch.tensor(state_dict['ens_args']['gain']))
    model.pop.bias.copy_(torch.tensor(state_dict['ens_args']['bias']))
    model.pop.d.copy_(torch.tensor(state_dict['conn_args']['weights']))

    state_dict = n3ml.to_state_dict_fpga(dt=state_dict['sim_args']['dt'],
                                         lr=state_dict['conn_args']['learning_rate'],
                                         model=model)

    print(state_dict)
    print()

    # n3ml.save(state_dict, mode='fpga', f=opt.save)
    n3ml.savez(state_dict, mode='fpga', f=opt.save, protocol=2)

    # state_dict = n3ml.load(f=opt.save, mode='fpga', allow_pickle=True)

    print(state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 실제 실행을 위해서는 경로가 수정되어야 합니다.
    parser.add_argument('--npz', default='data/npz/fpen_args_2975099280.npz')
    parser.add_argument('--save', default='data/npz/n3ml_202111021632.npz')

    app(parser.parse_args())
