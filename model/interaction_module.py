import torch
from torch import nn


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
