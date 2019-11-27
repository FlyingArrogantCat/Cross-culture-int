import torch
import numpy as np
from new_model.agent import Agent, CultureSpace
from .interaction_module import InteractionModel, StaticInteractionModel
from .demography_module import DemographyEnginer
from model.loss import InteractionLoss


class MainEngine(torch.nn.Module):
    def __init__(self, e_level=None, n_elements=100, size=100, threshold=0.01, birth=0.01, death=0.015):
        super(MainEngine, self).__init__()
        self.size = size
        self.n_elements = int(n_elements)
        self.threshold = threshold

        self.e_level = e_level if e_level is not None else [1]
        self.constant = None
        self.n_act_el = None

        self.interaction_model = InteractionModel(step=1e-1, size=self.size)
        self.list_obj = [Agent(size=size, e_level=1) for x in range(n_elements)]
        for obj in self.list_obj:
            obj.age = np.random.randint(0, 100)

        self.birth = birth
        self.death = death
        self.demography = DemographyEnginer(scale_b=self.birth, scale_d=self.death)
        self.culture_space = None
        self.fertility = None
        self.list_cult = None
        self.list_class = None
        self.amt = None

        self.model_loss = InteractionLoss()
        self.interaction_optimizer = torch.optim.Adam(self.interaction_model.params(), lr=1e-5)
        self.interaction_model = StaticInteractionModel(step=1e-1, size=self.size)

    def scenario(self, list_cult=None, list_amt=None, list_class=None, list_education=None, list_fertility=None,
                 depth_memory=100, give_mem_child=False):

        if list_cult is not None and list_amt is not None and list_education is not None:
            assert len(list_amt) == len(list_cult) == len(list_education) == len(list_fertility), "Corr Error"

            self.list_cult = list_cult
            self.list_class = list_class
            self.amt = list_amt

            self.e_level = list_education
            self.list_obj = []

            for indx, amt in enumerate(list_amt):
                for i in range(int(amt)):
                    self.list_obj.append(Agent(size=self.size, e_level=self.e_level[indx], cult_cond=list_cult[indx],
                                                self_class=list_class[indx], depth_memory=depth_memory))
        self.n_elements = len(self.list_obj)
        if list_fertility is not None:
            self.fertility = list_fertility
            self.demography = DemographyEnginer(scale_b=self.birth, scale_d=self.death, culture_fertility=self.fertility)
        if give_mem_child:
            self.demography = DemographyEnginer(scale_b=self.birth, scale_d=self.death,
                                                give_memory_child=give_mem_child)
        if give_mem_child and list_fertility is not None:
            self.demography = DemographyEnginer(scale_b=self.birth, scale_d=self.death,
                                                culture_fertility=self.fertility, give_memory_child=give_mem_child)

        train_classifier_x = [x.culture_condition.numpy() for x in self.list_obj]
        train_classifier_y = [x.sclass for x in self.list_obj]
        self.culture_space = CultureSpace(cultures=train_classifier_x, culture_classes=train_classifier_y)

    def step(self, indx, constant=300, energy=0):
        for obj in self.list_obj:
            obj.forward_memory()
            obj.forward_age()
            obj.sclass = self.culture_space.predict_culture([obj.culture_condition.detach().numpy()])[0]

        self.demography(self.list_obj)

        for x in self.list_obj:
            x.curr_energy = torch.tensor(energy)
        #        x.curr_energy = torch.FloatTensor(1).uniform_(0, energy)
        self.n_elements = len(self.list_obj)
        self.constant = int(constant)
        if self.constant > self.n_elements:
            self.constant = int(self.n_elements)

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

            result = self.interaction_model(acted.culture_condition * acted.curr_energy,
                                            action.culture_condition * action.curr_energy)

            if indx < 0:
                main_loss = self.model_loss(result, (acted.culture_condition,
                                                     action.culture_condition))

                main_loss.backward(retain_graph=True)
                self.interaction_optimizer.step()

            if indx >= 0:
                acted.culture_condition = result[0]
                action.culture_condition = result[1]

    def define_action_index(self):

        res = np.unique(np.random.randint(0, len(self.list_obj), self.constant))
        res = res.reshape((int(res.shape[0]/2), 2)) if res.shape[0] % 2 == 0 else res[:-1].reshape((int((res.shape[0] - 1)/2), 2))
        return res
