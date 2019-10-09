import torch
import numpy as np
from .base import Object, CultureSpace
from .interaction_module import InteractionModel
from .demography_module import DemographyEnginer
from .loss import InteractionLoss


class MainEngine(torch.nn.Module):
    def __init__(self, e_level=None, n_elements=100, size=100, threshold=0.01, birth=0.1, death=0.1):
        super(MainEngine, self).__init__()
        self.size = size
        self.n_elements = int(n_elements)
        self.threshold = threshold

        self.e_level = e_level if e_level is not None else [1]
        self.constant = None
        self.n_act_el = None

        self.interaction_model = InteractionModel(step=1e-3, size=self.size)
        self.list_obj = [Object(size=size, e_level=1) for x in range(n_elements)]
        for obj in self.list_obj:
            obj.age = np.random.randint(0, 100)

        self.birth = birth
        self.death = death
        self.demography = DemographyEnginer(scale_b=self.birth, scale_d=self.death)
        self.culture_space = None
        self.fertility = None

        self.model_loss = InteractionLoss()
        self.interaction_optimizer = torch.optim.Adam(self.interaction_model.params(), lr=1e-5)

    def scenario(self, list_cult=None, list_amt=None, list_class=None, list_education=None, list_fertility=None,
                 depth_memory=100):

        if list_cult is not None and list_amt is not None and list_education is not None:

            assert len(list_amt) == len(list_cult) == len(list_education), "Corr Error"

            self.e_level = list_education
            self.list_obj = []

            for indx, amt in enumerate(list_amt):
                for i in range(int(amt)):
                    self.list_obj.append(Object(size=self.size, e_level=self.e_level[indx], cult_cond=list_cult[indx],
                                                self_class=list_class[indx], depth_memory=depth_memory))

        if list_fertility is not None:
            self.fertility = list_fertility
            self.demography = DemographyEnginer(scale_b=self.birth, scale_d=self.death, culture_fertility=self.fertility)

        self.culture_space = CultureSpace(cultures=list_cult, culture_classes=list_class)

    def step(self, constant=100, energy=0, vecs=None):

        if vecs is not None:
            for obj in self.list_obj:
                obj.sclass(vecs[0], vecs[1])
        self.demography(self.list_obj)

        for x in self.list_obj:
                x.curr_energy = torch.FloatTensor(1).uniform_(0, energy)

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

            result = self.interaction_model(acted.culture_condition * acted.curr_energy,
                                            action.culture_condition * action.curr_energy)

            main_loss = self.model_loss(result, (acted.culture_condition,
                                                 action.culture_condition))

            main_loss.backward(retain_graph=True)
            self.interaction_optimizer.step()

            acted.culture_condition = acted.education * result[0]
            action.culture_condition = action.education * result[1]

        for obj in self.list_obj:
            obj.forward_memory()
            obj.forward_age()
            #obj.sclass = self.culture_space.predict_culture([obj.culture_condition.detach().cpu().numpy()])[0]

    def define_action_index(self):
        res = np.random.randint(0, len(self.list_obj), (self.constant, 2)).tolist()
        return res
