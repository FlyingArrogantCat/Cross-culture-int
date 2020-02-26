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

        self.birth_scale = 0
        self.death_scale = 0
        self.death_iter_border = 1
        self.death_at_birth = 0

        self.death_curve = lambda x: (1 - self.death_at_birth) * x / self.death_iter_border + self.death_at_birth
        self.birth_curve = lambda x: True if 14 < x < 70 else False

    def define_demography(self, scale_b=0.1, scale_d=0.1, death_iter_border=100):
        self.birth_scale = scale_b
        self.death_scale = scale_d
        self.death_iter_border = death_iter_border
        self.death_at_birth = 0.005

        self.death_curve = lambda x: (1 - self.death_at_birth) * x / self.death_iter_border + self.death_at_birth
        self.birth_curve = lambda x: True if 14 < x < 70 else False

    def initialize_experiment(self, cultures, member_amt, bases, critical_angles, mu=1):
        self.cultures = cultures
        self.cult_member_amt = member_amt
        self.culture_bases = bases
        self.critical_angles = critical_angles
        self.mu = mu

        for i in range(0, len(self.cult_member_amt)):
            for j in range(0, self.cult_member_amt[i]):
                self.agents.append(Agent(bases[i] + (bases[i] * np.random.normal(0, 0.1, self.n) /
                                                     np.linalg.norm(bases[i])), cultures[i],
                                         self.n,
                                         age=np.random.randint(14,30)))

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
                if self.angle_vec(self.agents[i].culture_state, base) < angle:
                    self.agents[i].culture = culture

    def clusters_factorization(self, clusters_list):
        mean_cultures_state = np.array([np.mean([self.agents[clusters_list[j][i]].culture_state for i in
                                                 range(len(clusters_list[j]))], axis=0) for j in range(len(clusters_list))])
        noticed_indx_list = []
        pairs = []
        free_indexs_list = [i for i in range(0, mean_cultures_state.shape[0])]

        #TODO
        # Add one more regime for randomly choosing the pairs
        while len(free_indexs_list) > 1:
            i = free_indexs_list[0]
            noticed_indx_list.append(i)
            state = mean_cultures_state[i]

            min_val = 10
            indx = 0
            for j in range(0, mean_cultures_state.shape[0]):
                if j not in noticed_indx_list:
                    val = self.angle_vec(state, mean_cultures_state[j])
                    if val < min_val:
                        indx = j
                        min_val = val
            pairs.append([i, indx])
            noticed_indx_list.append(indx)
            free_indexs_list = [x for x in free_indexs_list if x not in noticed_indx_list]

        return pairs, mean_cultures_state

    def do_demography_iteration(self):
        #TODO
        # Add one more regime for randomly choosing the pairs
        for obj in self.agents:
            obj.age += 1

        pairs = []
        noticed_indx_list = []
        free_indexs_list = [i for i in range(0, len(self.agents))]
        while len(free_indexs_list) > 1:
            i = free_indexs_list[0]
            noticed_indx_list.append(i)

            if not self.birth_curve(self.agents[i].age):
                free_indexs_list = [x for x in free_indexs_list if x not in noticed_indx_list]
                continue

            min_val = 10
            indx = 0
            for j in range(0, len(self.agents)):
                if j not in noticed_indx_list:
                    val = self.angle_vec(self.agents[i].culture_state, self.agents[j].culture_state)
                    if val < min_val:
                        indx = j
                        min_val = val
            pairs.append([i, indx])
            noticed_indx_list.append(indx)
            free_indexs_list = [x for x in free_indexs_list if x not in noticed_indx_list]

        num_dem_interaction = int(self.birth_scale * len(pairs)) + 1
        birth_index = np.random.randint(0, len(pairs), num_dem_interaction)

        for indx in birth_index:
            pair = pairs[indx]
            self.agents.append(Agent(culture_state=(self.agents[pair[0]].culture_state +
                                                    self.agents[pair[1]].culture_state)/2,
                                     culture=(self.agents[pair[0]].culture if np.random.uniform(0, 1) >
                                              0.5 else self.agents[pair[1]].culture),
                                     n=self.n)
                               )

        num_death = int(self.death_scale * len(self.agents))
        amt_death = 0
        del_indexs = []
        index_list = np.arange(len(self.agents))
        np.random.shuffle(index_list)

        for indx in index_list:
            if np.random.uniform(0, 1) < self.death_curve(self.agents[indx].age):
                del_indexs.append(indx)
                amt_death += 1
            if amt_death >= num_death:
                break

        del_indexs = - np.sort(-np.unique(del_indexs))
        del_indexs = del_indexs[:-1]

        temp_agents = deepcopy(self.agents)

        for indx in range(len(del_indexs)):
            del temp_agents[del_indexs[indx]]

        self.agents = deepcopy(temp_agents)

    def power_iteration(self):
        for obj in self.agents:
            np.append(obj.culture_memory, obj.culture_state)
        critical_angle = 0.1
        list_clusters = self.clusterization(self.agents, critical_angle)
        pairs, mean_cultures_state = self.clusters_factorization(list_clusters)
        interaction_angle = 0.1
        for pair in pairs:
            state_1 = mean_cultures_state[pair[0]]
            state_2 = mean_cultures_state[pair[1]]
            #TODO
            # include the memory
            new_1 = state_1 + np.cos(interaction_angle + self.angle_vec(state_1, state_2)) * (state_2 - state_1)
            new_2 = state_2 + np.cos(interaction_angle + self.angle_vec(state_1, state_2)) * (state_1 - state_2)

            for indx_obj in list_clusters[pair[0]]:
                self.agents[indx_obj].culture_state = (self.agents[indx_obj].culture_state + new_1) / \
                                                      np.linalg.norm(self.agents[indx_obj].culture_state + new_1)
            for indx_obj in list_clusters[pair[1]]:
                self.agents[indx_obj].culture_state = (self.agents[indx_obj].culture_state + new_2) / \
                                                      np.linalg.norm(self.agents[indx_obj].culture_state + new_2)

        self.do_demography_iteration()

        self.culture_rechanging()

    @staticmethod
    def angle(obj1, obj2):
        return np.arccos(np.dot(obj1.culture_state, obj2.culture_state) / (np.linalg.norm(obj1.culture_state)
                                                                           * np.linalg.norm(obj2.culture_state)))

    @staticmethod
    def angle_vec(vec1, vec2):
        return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
