import torch
from torch import nn
import numpy as np


class Object:
    def __init__(self, size=1024, e_level=0.5, cult_cond=None, feelings=None, depth_memory=100):
        self.size = size
        self.curr_energy = -1
        self.education = np.random.uniform(0, e_level)
        if cult_cond is None:
            self.culture_condition = np.random.normal(0, 1, size)
        else:
            self.culture_condition = cult_cond
        if feelings is None:
            self.feelings = np.random.normal(0, 1, size)
        else:
            self.feelings = feelings

        self.depth_memory = depth_memory
        self.culture_memory = np.zeros(depth_memory, size)
        self.memory_indx = 0
        self.condition = 'numpy'

    def get_tensor_representation(self):
        if self.condition == 'numpy':
            self.curr_energy = torch.tensor(self.curr_energy)
            self.education = torch.tensor(self.education)
            self.culture_condition = torch.from_numpy(self.culture_condition).float()
            self.feelings = torch.from_numpy(self.feelings).float()
            self.condition = 'torch'

    def get_numpy_representation(self):
        if self.condition == 'torch':
            self.curr_energy = self.curr_energy.detach().cpu().numpy()
            self.education = self.education.detach().cpu().numpy()
            self.culture_condition = self.culture_condition.detach().cpu().numpy()
            self.feelings = self.feelings.detach().cpu().numpy()
            self.condition = 'numpy'

    def forward_memory(self):
        if self.memory_indx+1 > self.depth_memory:
            self.mem_upd()
            self.memory_indx = self.depth_memory - 1
        self.culture_memory[self.memory_indx] = self.culture_condition

    def mem_upd(self):
        for indx in range(self.depth_memory - 1):
            self.culture_memory[indx] = self.culture_memory[indx + 1]

    @staticmethod
    def get_noise(size):
        return np.random.normal(0, 1, size)


class InteractionModel(nn.Module):
    def __init__(self, step=1e-5, size=1024):
        super(InteractionModel, self).__init__()
        self.step = step
        self.size = size
        self.sigmoid_1 = nn.Sequential(nn.Linear(self.size, self.size),
                                       nn.Sigmoid(),
                                       nn.Linear(self.size, self.size),
                                       nn.Sigmoid(),
                                       nn.Linear(self.size, self.size),
                                       nn.Sigmoid())

        self.sigmoid_2 = nn.Sequential(nn.Linear(self.size, self.size),
                                       nn.Sigmoid(),
                                       nn.Linear(self.size, self.size),
                                       nn.Sigmoid(),
                                       nn.Linear(self.size, self.size),
                                       nn.Sigmoid())

        self.sigmoid_3 = nn.Sequential(nn.Linear(self.size, self.size),
                                       nn.Sigmoid(),
                                       nn.Linear(self.size, self.size),
                                       nn.Sigmoid(),
                                       nn.Linear(self.size, self.size),
                                       nn.Sigmoid())

        self.tanh = nn.Sequential(nn.Linear(self.size, self.size),
                                  nn.Tanh(),
                                  nn.Linear(self.size, self.size),
                                  nn.Tanh(),
                                  nn.Linear(self.size, self.size),
                                  nn.Tanh())

    def forward(self, acted, action):
        first_l = acted * self.sigmoid_1(action)
        second_l = self.sigmoid_2(action) * self.tanh(action) + first_l
        acted_out_up = second_l
        action_out_up = torch.tanh(acted_out_up) * self.sigmoid_3(action)

        acted_out = acted + self.step * acted_out_up
        action_out = action + self.step * action_out_up

        return acted_out, action_out, acted_out_up, action_out_up

    def params(self):
        return [{'params': self.sigmoid_1.parameters()},
                {'params': self.sigmoid_2.parameters()},
                {'params': self.sigmoid_3.parameters()},
                {'params': self.tanh.parameters()}]


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

    def params(self):
        return [{'params': self.net.parameters()}]


class DemographyEnginer(nn.Module):
    def __init__(self, scale_b=0.2, scale_d=0.2):
        super(DemographyEnginer, self).__init__()
        self.existing_scale = scale_b
        self.death_scale = scale_d

    def forward(self, objs):
        for indx in range(np.round(np.random.uniform(0, self.existing_scale) * len(objs))):
            new_obj = Object(size=objs[indx].size, cult_cond=objs[indx].culture_condition, feelings=objs[indx].feelings)
            objs.append(new_obj)

        del_indexs = np.random.randint(0, len(objs), (np.round(np.random.uniform(0, self.existing_scale) * len(objs))))

        del_indexs = np.sort(del_indexs)

        for indx in range(len(del_indexs), -1, -1):
            del objs[del_indexs[indx]]
        return objs
