import torch
from torch import nn
import numpy as np
from .base import Object, InteractionModel, EnergyDecoder
from .loss import InteractionLoss, EnergyDecoderLoss


class MainEngine(torch.nn.Module):
    def __init__(self, e_level=0.5, n_elements=100, size=100, threshold=0.01):
        super(MainEngine, self).__init__()
        self.size = size
        self.n_elements = int(n_elements)
        self.threshold = threshold
        self.constant = None
        self.n_act_el = None

        self.interaction_model = InteractionModel(step=1e-3, size=self.size)
        self.feelings_model = InteractionModel(step=1e-2, size=self.size)
        self.decoder = EnergyDecoder(size=self.size)
        self.list_obj = [Object(size=size, e_level=e_level) for x in range(n_elements)]

        self.decoder_loss = EnergyDecoderLoss()
        self.model_loss = InteractionLoss()

        self.feelings_optimizer = torch.optim.Adam(self.feelings_model.params(), lr=1e-4)
        self.interaction_optimizer = torch.optim.Adam(self.interaction_model.params(), lr=1e-5)

    def step(self, constant=100, energy=0, update=None):
        for x in self.list_obj:
            if x.condition == 'numpy':
                x.curr_energy = np.random.uniform(0, energy)
            else:
                x.curr_energy = torch.FloatTensor(1).uniform_(0, energy)

        if update is not None:
            if self.list_obj[0].condition == 'numpy':
                self.list_obj[0].curr_energy = update
            else:
                self.list_obj[0].curr_energy = torch.tensor(update)

        self.constant = int(constant)
        if self.constant > self.n_elements:
            self.constant = int(self.n_elements / 2)
        self.n_act_el = np.random.randint(0, self.n_elements, size=self.constant)

        indexs = self.define_action_index()

        for index in indexs:
            self.interaction_optimizer.zero_grad()

            flag = True if np.random.uniform(0, 1) > 0.5 else False
            if flag:
                action = self.list_obj[index[0]]
                acted = self.list_obj[index[1]]
            else:
                action = self.list_obj[index[1]]
                acted = self.list_obj[index[0]]

            action.get_tensor_representation()
            acted.get_tensor_representation()

            result = self.interaction_model(acted.culture_condition * acted.curr_energy,
                                            action.culture_condition * action.curr_energy)

            main_loss = self.model_loss(result, (acted.culture_condition,
                                                 action.culture_condition))
            main_loss.backward(retain_graph=True)
            self.interaction_optimizer.step()

            acted.culture_condition = acted.education * result[0]
            action.culture_condition = action.education * result[1]

        for obj in self.list_obj:
            self.feelings_optimizer.zero_grad()

            obj.get_tensor_representation()

            result = self.feelings_model(obj.feelings * obj.curr_energy,
                                         obj.culture_condition * obj.curr_energy)

            main_loss = self.model_loss(result, (obj.feelings,
                                                 obj.culture_condition))
            main_loss.backward(retain_graph=True)
            self.feelings_optimizer.step()
            obj.feelings = obj.education * result[0]

    def define_action_index(self):

        for x in self.list_obj:
            x.get_numpy_representation()

        Cult_corr = np.zeros((self.constant, self.constant))
        Feel_corr = np.zeros((self.constant, self.constant))

        for i in range(self.constant):
            for j in range(self.constant):
                Cult_corr[i, j] = np.dot(self.list_obj[i].culture_condition, self.list_obj[j].culture_condition.T) / \
                                  np.dot(self.list_obj[i].culture_condition, self.list_obj[i].culture_condition.T)
                Feel_corr[i, j] = np.dot(self.list_obj[i].feelings, self.list_obj[j].feelings.T) / \
                                  np.dot(self.list_obj[i].feelings, self.list_obj[i].feelings.T)

        Cult_corr = (np.abs(Cult_corr) * np.abs(Feel_corr)) > self.threshold

        main_dict = {}
        for i in range(Cult_corr.shape[0]):
            for j in range(i, Cult_corr.shape[1]):
                if Cult_corr[i, j]:
                    main_dict[i] = j

        res = [[x, y] for x, y in zip(main_dict.keys(), main_dict.values())]
        return res
