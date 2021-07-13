from typing import Dict

import sklearn.datasets

import torch
import torchvision


class BaseDataLoader:
    def __init__(self):
        pass


class MNISTDataLoader(BaseDataLoader):
    def __init__(self, data, batch_size=1, shuffle=True, train=True):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                data,
                train=train,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
            batch_size=batch_size,
            shuffle=shuffle)
        self.it = iter(self.loader)
        self.o = next(self.it)

    def next(self):
        self.o = next(self.it)

    def __call__(self):
        return self.o


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self):
        pass


class DataLoader:
    def __init__(self):
        pass


class IRISDataLoader(DataLoader):
    def __init__(self, ratio: float = 0.8) -> None:
        super().__init__()
        self.ratio = ratio
        self.raw = sklearn.datasets.load_iris()
        self.index = None
        self.data = {}
        self.summary = {}

    def run(self) -> Dict[str, torch.Tensor]:
        data = torch.tensor(self.raw['data'])
        target = torch.tensor(self.raw['target'])

        index = torch.randperm(data.size(0))
        data = data[index]
        target = target[index]

        mid = int(data.size(0) * self.ratio)
        self.data['train.data'] = data[:mid]
        self.data['train.target'] = target[:mid]
        self.data['test.data'] = data[mid:]
        self.data['test.target'] = target[mid:]

        return self.data

    def summarize(self) -> Dict[str, torch.Tensor]:
        data = self.data['train.data']
        n = data.size(0)
        m = data.size(1)
        d_min = []
        d_max = []
        for j in range(m):
            min_val = data[0][j]
            max_val = data[0][j]
            for i in range(1, n):
                if data[i][j] < min_val:
                    min_val = data[i][j]
                if data[i][j] > max_val:
                    max_val = data[i][j]
            d_min.append(min_val)
            d_max.append(max_val)
        self.summary['min'] = torch.tensor(d_min)
        self.summary['max'] = torch.tensor(d_max)
        return self.summary
