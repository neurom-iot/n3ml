from typing import Dict

import numpy as np

import n3ml.network


def to_state_dict(dt: float, lr: float, model: n3ml.network.Network) -> Dict[str, Dict]:
    # 현재, 이 함수는 고정된 구조를 가지는 NEF 네트워크로부터 state dictionary를 추출
    # Network 구조는 고정되어 있는 것으로 봄.
    # 그 구조는 .npz 파일 형식을 따름.
    # 입력된 model 구조가 해당 형식에 맞는지 확인하지는 않음.
    sim_args = {'dt': dt}
    ens_args = {}
    conn_args = {'learning_rate': lr}
    recur_args = {}
    for m in model.population.values():
        if isinstance(m, n3ml.population.NEF):
            ens_args['input_dimensions'] = m.input_size
            ens_args['output_dimensions'] = m.output_size
            ens_args['n_neurons'] = m.neurons
            # TODO: torch.Tensor를 numpy.ndarray로 변환하기 위해서는 Tensor().detach().cpu().numpy()를
            #       항상 사용해야 하는가? 상황에 따라 다르게 사용해야 한다면 어떤 경우들이 있을까?
            #       그리고 상황별로 구분해서 구현해야 할 필요가 있을까?
            ens_args['bias'] = m.bias.detach().cpu().numpy()
            ens_args['gain'] = m.a.detach().cpu().numpy()
            ens_args['scaled_encoders'] = m.e.detach().cpu().numpy()
            # TODO: n3ml에서 사용하고 있는 neuron_type과 nengo에서 제공하는 neuron_type을 맞출 필요가 있다.
            ens_args['neuron_type'] = 'SpikingRectifiedLinear'
            conn_args['weights'] = m.d.detach().cpu().numpy()
            recur_args['weights'] = 0
            return {'sim_args': sim_args, 'ens_args': ens_args, 'conn_args': conn_args, 'recur_args': recur_args}
    raise Exception("Invalid model, expected model has a n3ml.population.NEF")


def save(obj: Dict[str, Dict], f: str) -> None:
    if not f.endswith('.npz'):
        f = f + '.npz'
    np.savez(f, obj)
