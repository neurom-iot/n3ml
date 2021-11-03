from typing import Any

import numpy as np

import torch.nn as nn

import n3ml.network


def to_state_dict_fpga(dt: float, lr: float, model: n3ml.network.Network):
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


def to_state_dict_loihi(model: n3ml.network.Network):
    state_dict_conv = [l for l in model.named_children() if isinstance(l[1], nn.Conv2d)]
    state_dict_linear = [l for l in model.named_children() if isinstance(l[1], nn.Linear)]

    state_dict = [l for l in reversed(state_dict_conv + [('', 0)] + state_dict_linear)]
    state_dict = {'arr_'+str(i): (np.array(state_dict[i][1], dtype=object) if isinstance(state_dict[i][1], int) else np.array(state_dict[i][1].weight.detach().cpu().numpy(), dtype=object)) for i in range(len(state_dict))}

    # PyTorch and Keras have different order of shape in convolutional layer
    for k in state_dict:
        if len(state_dict[k].shape) > 2:
            out_channels, in_channels, height, width = state_dict[k].shape  # in torch, conv's shape orders
            tmp = np.zeros((height, width, in_channels, out_channels))
            for i1 in range(height):
                for i2 in range(width):
                    for i3 in range(in_channels):
                        for i4 in range(out_channels):
                            tmp[i1, i2, i3, i4] = state_dict[k][i4, i3, i1, i2]
            state_dict[k] = tmp

    return state_dict


def save(state_dict: Any, mode: str, f: str) -> None:
    if not f.endswith('.npz'):
        f = f + '.npz'
    if mode == 'fpga' or mode == 'loihi':
        np.savez_compressed(f, **state_dict)
        return
    raise ValueError("Expected '{}' or '{}', but got '{}'".format('fpga', 'loihi', mode))


def savez(state_dict: Any, mode: str, f: str, protocol=2) -> None:
    if not f.endswith('.npz'):
        f = f + '.npz'
    import zipfile
    from numpy.lib.npyio import zipfile_factory
    from numpy.lib.format import write_array
    import pickle
    zipf = zipfile_factory(f, mode='w', compression=zipfile.ZIP_DEFLATED)
    for key, val in state_dict.items():
        fname = key + '.npy'
        val = np.asanyarray(val)
        with zipf.open(fname, 'w') as fid:
            # write_array(fid, val, allow_pickle=True)
            pickle.dump(val, fid, protocol=protocol)


def load(f: str, mode: str, allow_pickle: bool = True) -> Any:
    if not f.endswith('.npz'):
        f = f + '.npz'
    npz = np.load(f, allow_pickle=allow_pickle)
    if mode in ['pynq', 'de1-soc', 'loihi', 'fpga']:
        state_dict = {}
        for item in npz:
            if mode == 'fpga':  # TODO: Separate into 'pynq' and 'de1-soc'
                state_dict[item] = npz[item].tolist()
            elif mode == 'loihi':
                state_dict[item] = np.array(npz[item])
        return state_dict
    raise ValueError("Expected '{}' or '{}', but got '{}'".format('fpga', 'loihi', mode))


def _load(f: str, mode: str, allow_pickle: bool = True) -> Any:
    if not f.endswith('.npz'):
        f = f + '.npz'
    npz = np.load(f, protocol=2, allow_pickle=allow_pickle)
    if mode in ['fpga', 'loihi']:
        state_dict = {}
        for item in npz:
            if mode == 'fpga':
                state_dict[item] = npz[item].tolist()
            elif mode == 'loihi':
                state_dict[item] = np.array(npz[item])
        return state_dict
    raise ValueError("Expected '{}' or '{}', but got '{}'".format('fpga', 'loihi', mode))
