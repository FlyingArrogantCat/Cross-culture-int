import torch
import numpy as np
from sklearn import linear_model


class Object:
    def __init__(self, size=1024, e_level=0.5, cult_cond=None, depth_memory=100, self_class=0):
        '''indoor params'''
        self.size = size

        '''culture params'''
        self.curr_energy = torch.tensor(-1)
        self.sclass = self_class
        self.education = torch.tensor(e_level + np.random.normal(0, 0.2)).float()

        if cult_cond is None:
            self.culture_condition = torch.from_numpy(np.random.normal(0, 1, size)).float()
        else:
            self.culture_condition = torch.from_numpy(cult_cond).float()

        self.depth_memory = depth_memory
        if self.depth_memory != 0:
            self.culture_memory =torch.from_numpy(np.zeros((depth_memory, size))).float()
        else:
            self.culture_memory = None
        self.memory_indx = 0

        '''demography params'''
        self.age = 0

    def forward_memory(self):
        if self.depth_memory == 0:
            return 0
        if self.memory_indx + 1 > self.depth_memory - 1:
            self.mem_upd()
            self.memory_indx = self.depth_memory - 1
        else:
            self.memory_indx = self.memory_indx + 1

        self.culture_memory[self.memory_indx] = self.culture_condition
        self.remind_memory()

    def remind_memory(self, param=0.5):
        if np.random.uniform(0, 1) > param:
            return 0
        else:
            self.culture_condition = self.culture_memory[np.random.randint(0, self.memory_indx)]

    def mem_upd(self):
        for indx in range(self.depth_memory - 1):
            self.culture_memory[indx] = self.culture_memory[indx + 1]

    def forward_age(self):
        self.age += 1

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


class CultureSpace:
    def __init__(self, cultures, culture_classes):
        self.cultures = cultures
        self.classes = culture_classes

        self.classifier = linear_model.SGDClassifier(max_iter=50)
        self.classifier.fit(self.cultures, self.classes)

    def predict_culture(self, cult_sample):
        return self.classifier.predict(cult_sample)