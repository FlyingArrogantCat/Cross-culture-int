import torch
from torch import nn
import numpy as np


class Object:
    def __init__(self, size=1024, e_level=0.5):
        self.size = size
        self.curr_energy = -1
        self.education = np.random.uniform(0, e_level)
        self.culture_condition = np.random.normal(0, 1, size)
        self.condition = 'numpy'

    def get_tensor_representation(self):
        if self.condition == 'numpy':
            self.curr_energy = torch.from_numpy(self.curr_energy).float()
            self.education = torch.from_numpy(self.education).float()
            self.culture_condition = torch.from_numpy(self.culture_condition).float()
            self.condition = 'torch'

    def get_numpy_representation(self):
        if self.condition == 'torch':
            self.curr_energy = self.curr_energy.detach().cpu().numpy()
            self.education = self.education.detach().cpu().numpy()
            self.culture_condition = self.culture_condition.deatch().cpu().numpy()
            self.condition = 'numpy'

    @staticmethod
    def get_noise(size):
        return np.random.normal(0, 1, size)


class InteractionModel(nn.Module):
    def __init__(self, step=1e-2, size=1024):
        super(InteractionModel, self).__init__()
        self.step = step
        self.size = size
        self.sigmoid_1 = nn.Sequential(nn.Linear(self.size, self.size),
                                       nn.Sigmoid())
        self.sigmoid_2 = nn.Sequential(nn.Linear(self.size, self.size),
                                       nn.Sigmoid())
        self.sigmoid_3 = nn.Sequential(nn.Linear(self.size, self.size),
                                       nn.Sigmoid())
        self.tanh = nn.Sequential(nn.Linear(self.size, self.size),
                                  nn.Tanh())

    def forward(self, acted, action):
        first_l = acted * self.sigmoid_1(action)
        second_l = self.sigmoid_2(action) * self.tanh(action) + first_l
        acted_out_up = second_l
        action_out_up = torch.tanh(acted_out_up) * self.sigmoid_3(action)

        acted_out = acted + self.step * acted_out_up
        action_out = action + self.step * action_out_up

        return acted_out, action_out, acted_out_up, action_out_up


class EnergyDecoder(nn.Module):
    def __init__(self, size=1000):
        super(EnergyDecoder, self).__init__()

        self.net = nn.Sequential(nn.Linear(1, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, size),
                                 nn.Sigmoid())

    def forward(self, x):
        out = self.net(x)
        return out