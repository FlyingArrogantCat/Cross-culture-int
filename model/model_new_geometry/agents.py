import torch
import numpy as np


class Agent:
    def __init__(self):
        self.age = 0
        self.k = 0
        self.m = 0
        self.n = self.k + self.m
        self.culture_state = np.random.normal(0, 1)
        self.culture_memory = np.array([])
        self.culture = 0

    def initializing(self, culture_state, culture):
        self.age = 0
        self.culture_state = culture_state + np.random.normal(0, 1)
        self.culture = culture

    def __print__(self):
        print('Agent: ')
        print('Age: ', self.age)
        print('Size of culture memory: ', self.culture_memory.shape())
        print('Culture state:', self.culture)
