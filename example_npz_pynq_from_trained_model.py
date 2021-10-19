"""
    데모를 고려한 예제로 N3ML을 사용해서 학습시킨 모델의 정보로부터 npz을 생성하는 시나리오를 설정하였습니다.
    현재, N3ML에서 구현된 PES 학습 알고리즘은 동작하지 않는 상태이기 때문에 기존에 학습된 정보를 포함하고 있는
    npz 파일을 N3ML에서 로드를 하고 이 정보를 기반으로 신경망을 생성합니다. 즉, 학습된 모델이 있다는 가정을
    한 후에 이 정보로부터 npz 파일을 생성하는 시나리오가 되겠습니다.
"""
import argparse

import numpy as np

import torch

import n3ml
from n3ml.model import Voelker2015


def app(opt):
    print(opt)

    npz = np.load(opt.pretrained, allow_pickle=True)

    state_dict = {}
    for item in npz:
        state_dict[item] = npz[item].tolist()

    model = Voelker2015(neurons=state_dict['ens_args']['n_neurons'],
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

    n3ml.save(state_dict, mode='fpga', f=opt.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 실제 실행을 위해서는 경로가 수정되어야 합니다.
    parser.add_argument('--pretrained', default='data/npz/fpen_args_2975099280.npz')
    parser.add_argument('--save', default='data/npz/n3ml_pynq_202110191514.npz')

    app(parser.parse_args())
