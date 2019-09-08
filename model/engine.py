import torch
from torch import nn
import numpy as np
from .base import Object, InteractionModel, EnergyDecoder
from .loss import InteractionLoss


class MainEngine(torch.nn.Module):
    def __init__(self, e_level=0.5, n_elements=100, size=100, threshold=0.1):
        super(MainEngine, self).__init__()
        self.size = size
        self.n_elements = n_elements
        self.threshold = threshold
        self.constant = None
        self.n_act_el = None

        self.interaction_model = InteractionModel(step=1e-3, size=self.size)
        self.decoder = EnergyDecoder(size=self.size)
        self.list_obj = [Object(size=size, e_level=e_level) for x in range(n_elements)]

    def step(self, constant=100, energy=0.5):
        for x in self.list_obj:
            x.curr_energy = np.random.uniform(0, energy)

        self.constant = constant
        if self.constant > self.n_elements:
            self.constant = int(self.n_elements / 2)
        self.n_act_el = np.random.randint(0, self.n_elements, size=self.constant)

        decoder_loss = nn.CrossEntropyLoss()
        model_loss = InteractionLoss()

        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-5)
        model_optimizer = torch.optim.Adam(self.interaction_model.parameters(), lr=1e-5)

        indexs = self.define_action_index()
        for index in indexs:
            flag = True if np.random.uniform(0, 1) > 0.5 else False
            if flag:
                action = self.list_obj[index[0]]
                acted = self.list_obj[index[1]]
            else:
                action = self.list_obj[index[1]]
                acted = self.list_obj[index[0]]

            action.get_tensor_representation()
            acted.get_tensor_representation()

            vec_energy_acted = self.decoder(acted.curr_energy)
            vec_energy_action = self.decoder(action.curr_energy)

            dec_loss = decoder_loss(acted.curr_energy, vec_energy_acted)
            dec_loss.backward()
            dec_loss = decoder_loss(acted.curr_energy, vec_energy_action)
            dec_loss.backward()
            decoder_optimizer.step()

            result = self.interaction_model(acted.culture_condition + vec_energy_acted,
                                            action.culture_condition + vec_energy_action)

            main_loss = model_loss(result, (acted.culture_condition, action.culture_condition))
            main_loss.backward()
            model_optimizer.step()

            acted.culture_condition = result[0]
            action.culture_condition = result[1]

    def define_action_index(self):
        K = np.zeros((self.constant, self.constant))

        for i in range(self.constant):
            for j in range(self.constant):
                 K[i, j] = self.list_obj[i].culture_condition * self.list_obj[j] / \
                           (self.list_obj[i].culture_condition * self.list_obj[i].culture_condition)

        K = np.abs(K) > self.threshold

        main_dict = {}
        for i in range(K.shape[0]):
            for j in range(i, K.shape[1]):
                if K[i, j]:
                    main_dict[i] = j

        return [[x, y] for x, y in (main_dict.keys(), main_dict.values())]
