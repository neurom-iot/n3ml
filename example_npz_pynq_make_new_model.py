"""
    새로운 신경망을 생성하는 경우에 대한 예제가 되겠습니다. 새로운 신경망을 만들 때 필요한 인자를 받은 후에
    해당 인자를 사용해서 신경망을 생성하게 됩니다. 이때 가중치는 랜덤한 방법으로 초기화를 진행하였습니다.
"""
import argparse

import n3ml
from n3ml.model import Voelker2015


def app(opt):
    # 이 예제에서 입력된 dt의 단위는 s로 가정을 하였습니다.
    model = Voelker2015(neurons=opt.n_neurons,
                        input_size=opt.input_dimensions,
                        output_size=opt.output_dimensions,
                        neuron_type=opt.neuron_type,
                        dt=opt.dt)

    state_dict = n3ml.to_state_dict_fpga(dt=opt.dt, lr=opt.learning_rate, model=model)

    n3ml.save(state_dict, mode='fpga', f=opt.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dt', default=0.001, type=float)
    parser.add_argument('--input_dimensions', default=196, type=int)
    parser.add_argument('--output_dimensions', default=10, type=int)
    parser.add_argument('--n_neurons', default=81, type=int)
    parser.add_argument('--neuron_type', default='SpikingRectifiedLinear', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)

    # 실제 실행을 위해서는 경로가 수정되어야 합니다.
    parser.add_argument('--save', default='data/npz/n3ml_de1-soc_20211019.npz')

    app(parser.parse_args())
