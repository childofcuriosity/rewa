import torch
import numpy as np
import random
from torchvision.transforms import ToTensor


class LogicRegDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=50, dim=100, transform=ToTensor(), seed=0):
        self.num_samples = num_samples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.dim = dim
        self.pts = np.random.normal(size=num_samples * dim).reshape(num_samples, -1)
        self.v = np.zeros(dim)
        self.v[0] = 1.
        self.v[1] = 0.5
        self.b = np.random.normal(size=num_samples)
        # print(self.pts.shape, self.v.shape)
        self.targets = 1. / (1. + np.exp(- np.matmul(self.pts, self.v)))
        # self.targets = np.sigmoid(np.matmul(self.pts, self.v))
        self.targets = 1. * (np.random.rand(num_samples) < self.targets)
        # self.targets = np.matmul(self.pts, self.v)
        # print(self.targets.shape)
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.pts[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)




class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=50, dim=100, transform=ToTensor(), seed=0):
        self.num_samples = num_samples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.dim = dim
        self.pts = np.random.normal(size=num_samples * dim).reshape(num_samples, -1)
        self.v = np.zeros(dim)
        self.v[0] = 1.
        self.b = np.random.normal(size=num_samples)
        # print(self.pts.shape, self.v.shape)
        self.targets = np.matmul(self.pts, self.v) + self.b
        # print(self.targets.shape)
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.pts[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)


