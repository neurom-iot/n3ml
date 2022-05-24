from typing import Type

import torch
import torch.nn as nn
import torch.distributions.distribution as distribution

import n3ml.population as population
import n3ml.learning as learning


class Synapse(nn.Module):
    def __init__(self,
                 source: population.Population,
                 target: population.Population,
                 w: torch.Tensor,
                 w_min: float = 0.0,
                 w_max: float = 1.0,
                 alpha: float = None,
                 learning_rule: Type[learning.LearningRule] = None,
                 initializer: torch.distributions.distribution.Distribution = None) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.register_buffer('w', w)
        self.w_min = w_min
        self.w_max = w_max
        self.alpha = alpha
        if learning_rule is None:
            self.learning_rule = learning_rule
        else:
            self.learning_rule = learning_rule(self)
        self.initializer = initializer

    def init(self) -> None:
        self.w[:] = self.initializer.sample(sample_shape=self.w.size())

    def normalize(self) -> None:
        if self.alpha is not None:
            w_abs_sum = self.w.abs().sum(dim=1).unsqueeze(dim=1)
            w_abs_sum[w_abs_sum == 0.0] = 1.0
            self.w *= self.alpha / w_abs_sum

    def update(self) -> None:
        if self.learning_rule is not None:
            self.learning_rule.run()

    def run(self) -> None:
        raise NotImplementedError


class LinearSynapse(Synapse):
    def __init__(self,
                 source: population.Population,
                 target: population.Population,
                 w: torch.Tensor = None,
                 w_min: float = 0.0,
                 w_max: float = 1.0,
                 alpha: float = None,
                 learning_rule: learning.LearningRule = None,
                 initializer: torch.distributions.distribution.Distribution = None) -> None:
        if w is None:
            w = torch.zeros(size=(target.neurons, source.neurons))
        super().__init__(source, target, w, w_min, w_max, alpha, learning_rule, initializer)

    def run(self) -> torch.Tensor:
        """
            Non batch processing
        
            self.w.size:        [self.target.neurons, self.source.neurons]
            self.source.s.size: [self.source.neurons]
        """
        return torch.matmul(self.w, self.source.s)


class ConvSynapse(Synapse):
    pass
