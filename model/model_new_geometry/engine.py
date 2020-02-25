import numpy as np

from .interaction import *
from .agent import *


class MainEngine:
    def __init__(self, k=1, m=1):
        self.cultures = []
        self.cult_member_amt = []
        self.culture_bases = []
        self.critical_angles = []
        self.agents = []
        self.n = k + m
        self.mu = 0

    def initialize_experiment(self, cultures, member_amt, bases, critical_angles, mu=1):
        self.cultures = cultures
        self.cult_member_amt = member_amt
        self.culture_bases = bases
        self.critical_angles = critical_angles
        self.mu = mu

        for i in range(0, len(self.cult_member_amt)):
            for j in range(0, self.cult_member_amt[i]):
                self.agents.append(Agent(bases[i] + (bases[i] * np.random.normal(0, 0.1, self.n) /
                                                     np.linalg.norm(bases[i])), cultures[i], self.n, age=0))

    def clusterization(self, list_obj, crit_angle=0.1):
        noticed_obj_list = []
        clusters_list = []
        free_indx_list = [i for i in range(0, len(list_obj))]

        while len(free_indx_list) > 0:
            obj = free_indx_list[0]
            temp_list = [free_indx_list[0]]

            temp_len = 0
            while len(temp_list) != temp_len:
                for i in free_indx_list:
                    if i not in temp_list and self.angle(list_obj[obj], list_obj[i]) <= crit_angle:
                        temp_list.append(i)
                temp_len = len(temp_list)
            clusters_list.append(temp_list)

            for i in range(len(temp_list)):
                noticed_obj_list.append(temp_list[i])
            free_indx_list = [x for x in free_indx_list if x not in noticed_obj_list]
            #print('kek')
        return clusters_list

    def culture_rechanging(self):
        for i in range(0, len(self.agents)):
            for angle, base, culture in zip(self.critical_angles, self.culture_bases, self.cultures):
                if angle(self.agents[i], base) < angle:
                    self.agents[i].culture = culture

    def interaction(self):
        mean_cultures_state = []

    @staticmethod
    def angle(obj1, obj2):
        return np.arccos(np.dot(obj1.culture_state, obj2.culture_state) / (np.linalg.norm(obj1.culture_state)
                                                                           * np.linalg.norm(obj2.culture_state)))



class DemographyEnginer:
    def __init__(self, scale_b=0.2, scale_d=0.2, death_iter_border=100):
        self.existing_scale = scale_b
        self.death_scale = scale_d
        self.death_iter_border = death_iter_border
        self.death_at_birth = 0.005

        self.dem_curve = lambda x: (1 - self.death_at_birth) * x / self.death_iter_border + self.death_at_birth
        self.birth_curve = lambda x: True if 14 < x < 70 else False

    def forward(self, objs):
        #TODO
        return objs
