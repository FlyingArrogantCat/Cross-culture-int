import torch
import numpy as np


class Agent:
    def __init__(self, culture_state, culture, n, age=0):
        self.age = age
        self.n = n
        self.culture_state = culture_state + np.random.normal(0, 1)
        self.culture_memory = np.array([])
        self.culture = culture

    def __print__(self):
        print('Agent: ')
        print('Age: ', self.age)
        print('Size of culture memory: ', self.culture_memory.shape())
        print('Culture state:', self.culture)
