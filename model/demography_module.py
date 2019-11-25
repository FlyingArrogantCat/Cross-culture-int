from torch import nn
import numpy as np
import copy
import torch
from .base import Object


class DemographyEnginer(nn.Module):
    def __init__(self, scale_b=0.2, scale_d=0.2, death_iter_border=100, culture_fertility=None, give_memory_child=True):
        super(DemographyEnginer, self).__init__()
        self.existing_scale = scale_b
        self.death_scale = scale_d
        self.death_iter_border = death_iter_border
        self.death_at_birth = 0.05
        self.fertility = culture_fertility

        self.dem_curve = lambda x: (1 - self.death_at_birth) * x / self.death_iter_border + self.death_at_birth
        self.birth_curve = lambda x: True if 14 < x < 70 else False
        self.give_memory_child = give_memory_child

    def forward(self, objs):

        if self.existing_scale == 0 and self.death_scale == 0:
            return objs
        lenn = len(objs)
        for indx in (np.random.randint(0, lenn, int(np.random.uniform(0, self.existing_scale) * lenn))):
            if self.birth_curve(objs[indx].age):
                if self.fertility is not None:
                    if np.random.uniform(0, 1) < self.fertility[objs[indx].sclass]:
                        new_obj = Object(size=objs[indx].size, e_level=objs[indx].education,
                                         depth_memory=objs[indx].depth_memory, self_class=objs[indx].sclass)
                        new_obj.age = 0
                        if self.give_memory_child:
                            if objs[indx].depth_memory != 0:
                                new_obj.culture_memory[:int(new_obj.depth_memory/2)] = \
                                    objs[indx].culture_memory[:int(objs[indx].depth_memory/2)]
                                new_obj.memory_indx = int(new_obj.depth_memory/2) - 1
                        new_obj.culture_condition = objs[indx].culture_condition.clone().detach()
                        new_obj.culture_condition.add_(torch.from_numpy(np.random.normal(0, 0.1, new_obj.size)).float())
                        objs.append(new_obj)
                else:
                    new_obj = Object(size=objs[indx].size, e_level=objs[indx].education,
                                     depth_memory=objs[indx].depth_memory, self_class=objs[indx].sclass)
                    new_obj.age = 0
                    if self.give_memory_child:
                        if objs[indx].depth_memory != 0:
                            new_obj.culture_memory[:int(new_obj.depth_memory / 2)] = \
                                objs[indx].culture_memory[:int(objs[indx].depth_memory / 2)]
                            new_obj.memory_indx = int(new_obj.depth_memory / 2) - 1
                    new_obj.culture_condition = objs[indx].culture_condition.clone().detach()
                    new_obj.culture_condition.add_(torch.from_numpy(np.random.normal(0, 0.1, new_obj.size)).float())
                    objs.append(new_obj)

        del_indexs = np.unique(np.random.randint(0, lenn, (int(np.random.uniform(0, self.death_scale) * lenn))))
        for indx in range(len(del_indexs)):
            if np.random.uniform(0, 1) < self.dem_curve(objs[del_indexs[indx]].age):
                del_indexs[indx] = 0
        del_indexs = - np.sort(-np.unique(del_indexs))
        del_indexs = del_indexs[:-1]

        for indx in range(len(del_indexs)):
                del objs[del_indexs[indx]]
        return objs
