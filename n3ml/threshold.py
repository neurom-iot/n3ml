from typing import Any, List

import torch
import torch.nn as nn

import n3ml.network as network


def activation_based(train_loader: Any, model: network.Network):
    """
    - This function implements Diehl2015 algorithm that finds the proper thresholds
    for ANN-SNN conversion.
    - The function assumes that the input model is a trained ANN.
    """
    model.eval()
    ths = []  # The number of learnable layers
    for m in model.named_children():
        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear) :
            ths.append(0.0)

    for it, (images, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            model(x= images, find_threshold = True, threshold= ths)
    return ths


def activation_based2(train_loader: Any, model: network.Network):
    """
    - This function implements Diehl2015 algorithm that finds the proper thresholds
    for ANN-SNN conversion.
    - The function assumes that the input model is a trained ANN.
    """
    model.eval()
    ths = []  # The number of learnable layers
    for m in model.named_children():
        if isinstance(m[1], nn.ReLU) or isinstance(m[1], nn.Linear) :
            ths.append(0.0)

    for it, (images, _) in enumerate(train_loader):
        layer_index = -1
        if torch.cuda.is_available():
            x = images.cuda()
        with torch.no_grad():
            for m in model.named_children():
                # Assume that snn is a sequential model
                x = m[1](x)
                if isinstance(m[1], nn.ReLU) or isinstance(m[1], nn.Linear):
                    layer_index+=1
                    ths[layer_index]= max(ths[layer_index], x.max())

    return ths


def current_channel_wise(train_loader: Any, model: network.Network, num_steps: int, scaling_factor: float=1.):
    """
    This function implements Input current-based channel-wise algorithm that finds the proper thresholds
    for ANN-SNN conversion. The function assumes that the input model is a SNN.
    """
    ths = []  # The number of learnable layers
    for m in model.named_children():
        if isinstance(m[1], nn.Conv2d):
            num_channels = m[1].out_channels
            ths.append([0.0]*num_channels)
        elif isinstance(m[1], nn.Linear):
            ths.append(0.0)

    #remove the last layer value
    ths = ths[:-1]

    print(model.conv1_if.threshold)
    for l in range(len(ths)):
        print(l)
        for it, (images, _) in enumerate(train_loader):
            model.init_neuron_models()
            with torch.no_grad():
                for t in range(num_steps):
                    p = 0
                    if torch.cuda.is_available():
                        x = images.cuda()
                    for m in model.named_children():
                        # Assume that snn is a sequential model
                        x = m[1](x)

                        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                            if p == l:
                                if isinstance(m[1], nn.Conv2d):
                                    sorted_matrix = torch.amax(x, 0)
                                    for i in range(len(ths[l])):
                                        ths[l][i] = max(ths[l][i], sorted_matrix[i].max())
                                    break
                                if isinstance(m[1], nn.Linear):
                                    ths[l] = max(ths[l], x.max())
                                    break
                            else:  # p < l
                                p += 1

        model.update_threshold(ths)

    return ths


def spikenorm(train_loader: Any,
              encoder: Any,
              model: network.Network,
              num_steps: int,
              scaling_factor: float) -> List[float]:
    """
    This function implements Spike Norm algorithm that finds the proper thresholds
    for ANN-SNN conversion. The function assumes that the input model is a SNN.
    """
    num_layers = 0  # The number of learnable layers
    for m in model.named_children():
        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
            num_layers += 1
    ths = [0.0] * num_layers
    for l in range(num_layers):
        for it, (images, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
            with torch.no_grad():
                max_mem = 0.0
                for t in range(num_steps):
                    x = encoder(images).float()
                    p = 0
                    for m in model.named_children():
                        # Assume that snn is a sequential model
                        x = m[1](x)
                        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                            if p == l:
                                max_mem = max(max_mem, x.max())
                                break
                            else:  # p < l
                                p += 1
                ths[l] = max(scaling_factor * ths[l], max_mem)
                # TODO: if 문을 어떻게 조절할 수 있을까?
                # TODO: upper_bound(it)은 여러 요인에 따라 달라질 수 있다.
                if it == 0:
                    break
        print("{}-th layer's threshold: {}".format(l, ths[l]))
        model.update_threshold([th for th in ths])
    print("the found thresholds: {}".format(ths))
    return ths
