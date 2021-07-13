from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

import n3ml.population
import n3ml.connection
import n3ml.layer


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.population = {}
        self.connection = {}
        self.layer = OrderedDict()

    def _add_population(self, name, population):
        self.population[name] = population

    def _add_connection(self, name, connection):
        self.connection[name] = connection

    def _add_layer(self, name, layer):
        self.layer[name] = layer

    def add_component(self, name, component):
        self.add_module(name, component)

        if isinstance(component, n3ml.population.Population):
            self._add_population(name, component)
        elif isinstance(component, n3ml.connection.Synapse):
            self._add_connection(name, component)
        elif isinstance(component, n3ml.layer.Layer):
            self._add_layer(name, component)

    """
        def init(self)처럼 바로 실행하도록 하려면 Distribution을 생성자로 넘기는 것이 좋겠다.
    """
    def init(self):
        pass

    def initialize(self, **kwargs) -> None:
        for l in self.named_children():
            l[1].initialize(**kwargs)

    def init_param(self) -> None:
        for p in self.population.values():
            p.init_param()

    def normalize(self) -> None:
        for c in self.connection.values():
            c.normalize()

    def update(self) -> None:
        # TODO: update()를 어떻게 해야 추상화 할 수 있을까?
        # TODO: non-BP 기반 학습 알고리즘은 update()를 사용하여 학습을 수행한다.
        for synapse in self.connection.values():
            synapse.update()

    def run(self, x: Dict[str, torch.Tensor]) -> None:
        input = {}
        for name in x:
            input[name] = x[name].clone()
        for p_name in self.population:
            if isinstance(self.population[p_name], n3ml.population.Input):
                if p_name not in input:
                    input[p_name] = torch.zeros(self.population[p_name].neurons)
            else:
                for c_name in self.connection:
                    if self.connection[c_name].target == self.population[p_name]:
                        if p_name in input:
                            input[p_name] += self.connection[c_name].run()
                        else:
                            input[p_name] = self.connection[c_name].run()

        for p_name in self.population:
            self.population[p_name].run(input[p_name])
