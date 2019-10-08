import torch
import numpy as np


class Object:
    def __init__(self, size=1024, e_level=0.5, cult_cond=None, depth_memory=100, self_class=0):
        '''indoor params'''
        self.condition = 'numpy'
        self.size = size

        '''culture params'''
        self.curr_energy = -1
        self.sclass = self_class
        self.education = np.random.uniform(0, e_level)
        if cult_cond is None:
            self.culture_condition = np.random.normal(0, 1, size)
        else:
            self.culture_condition = cult_cond
        self.depth_memory = depth_memory
        if self.depth_memory != 0:
            self.culture_memory = np.zeros((depth_memory, size))
        else:
            self.culture_memory = None
        self.memory_indx = 0

        '''demography params'''
        self.age = 0

    def get_tensor_representation(self):
        if self.condition == 'numpy':
            self.curr_energy = torch.tensor(self.curr_energy)
            self.education = torch.tensor(self.education)
            self.culture_condition = torch.from_numpy(self.culture_condition).float()
            self.condition = 'torch'

    def get_numpy_representation(self):
        if self.condition == 'torch':
            self.curr_energy = self.curr_energy.detach().cpu().numpy()
            self.education = self.education.detach().cpu().numpy()
            self.culture_condition = self.culture_condition.detach().cpu().numpy()
            self.condition = 'numpy'

    def forward_memory(self):
        if self.depth_memory == 0:
            return 0
        if self.memory_indx + 1 > self.depth_memory - 1:
            self.mem_upd()
            self.memory_indx = self.depth_memory - 1
        else:
            self.memory_indx = self.memory_indx + 1

        if self.condition == 'torch':
            self.culture_memory[self.memory_indx] = self.culture_condition.detach().cpu().numpy()
        else:
            self.culture_memory[self.memory_indx] = self.culture_condition
        self.remind_memory()

    def remind_memory(self, param=0.5):
        if np.random.uniform(0, 1) > param:
            return 0
        else:
            if self.condition == 'torch':
                self.culture_condition = torch.from_numpy(self.culture_memory[
                                                              np.random.randint(0, self.memory_indx)]).float()
            else:
                self.culture_condition = self.culture_memory[np.random.randint(0, self.memory_indx)]

    def mem_upd(self):
        for indx in range(self.depth_memory - 1):
            self.culture_memory[indx] = self.culture_memory[indx + 1]

    def sclass(self, vec1, vec2):
        if self.condition == 'numpy':
            vec = (vec1 + vec2) - self.culture_condition
        else:
            vec = (vec1 + vec2) - self.culture_condition.detach().cpu().numpy()
        norm1 = np.linalg.norm(vec1 - vec)
        norm2 = np.linalg.norm(vec2 - vec)
        if norm1 > norm2:
            self.sclass = 0
        if norm2 > norm1:
            self.sclass = 1

    def forward_age(self):
        self.age += 1


class CultureSpace:
    def __init__(self, cultures):
        self.cultures = cultures
