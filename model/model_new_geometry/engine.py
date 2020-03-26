import numpy as np

from .interaction import *
from .agent import *


class MainEngine:
    def __init__(self, k=1, m=1, step=0.1):
        self.step = step
        self.cultures = []
        self.cult_member_amt = []
        self.culture_bases = []
        self.critical_angles = []
        self.agents = []
        self.education = []
        self.fertility = []
        self.n = k + m
        self.mu = 0

        self.birth_scale = 0
        self.death_scale = 0

        self.graph_num_cluster_cross_culture = 0
        self.graph_list_num_cluster_unique_culture = []
        self.graph_list_std_per_culture = []
        self.graph_list_mean_per_culture = []

    def define_demography(self, scale_b=0.1, scale_d=0.1):
        self.birth_scale = scale_b
        self.death_scale = scale_d

    def initialize_experiment(self, cultures, member_amt, bases, critical_angles, education_scales, mu=1, fertility=None):
        self.cultures = cultures
        self.cult_member_amt = member_amt
        self.culture_bases = bases
        self.critical_angles = critical_angles
        self.education = education_scales
        self.mu = mu

        self.fertility = [1] * len(self.cultures)
        if fertility is not None:
            self.fertility = fertility

        for i in range(0, len(self.cult_member_amt)):
            for j in range(0, self.cult_member_amt[i]):
                new_vec = np.ones(self.n) + self.culture_bases[i]
                while self.angle_vec(self.culture_bases[i], new_vec) < self.critical_angles[i]:
                    new_vec = np.random.normal(1, 1, self.n)
                    new_vec /= np.linalg.norm(new_vec)
                self.agents.append(Agent(new_vec, cultures[i],
                                         self.n,
                                         age=0))

        self.graph_list_num_cluster_unique_culture = [0] * len(cultures)

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
        num_death = int(self.death_scale * len(self.agents))

        amt_agent_cult = [np.sum([1 for x in self.agents if x.culture == y]) for y in self.cultures]
        amt_born_cult = [x * y for x, y in zip(amt_agent_cult, self.fertility)]

        for x, y in zip(self.cultures, amt_born_cult):
            num_dem_interaction = int(self.birth_scale * y)
            pairs = []
            noticed_indx_list = []
            free_indexs_list = []
            for i in range(len(self.agents)):
                if self.agents[i].culture == x:
                    free_indexs_list.append(i)

            while len(free_indexs_list) > 1:
                i = free_indexs_list[0]
                noticed_indx_list.append(i)

                if self.agents[i].age == 1:
                    free_indexs_list = [x for x in free_indexs_list if x not in noticed_indx_list]
                    continue

                min_val = 100000
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

            birth_index = np.random.randint(0, len(pairs), num_dem_interaction)

            for indx in birth_index:
                pair = pairs[indx]
                self.agents.append(Agent(culture_state=(self.agents[pair[0]].culture_state +
                                                        self.agents[pair[1]].culture_state)/2,
                                         culture=(self.agents[pair[0]].culture if np.random.uniform(0, 1) >
                                                  0.5 else self.agents[pair[1]].culture),
                                         n=self.n,
                                         age=1)
                                   )
        amt_death = 0
        del_indexs = []
        index_list = np.arange(len(self.agents))
        np.random.shuffle(index_list)

        while amt_death < num_death:
            for indx in [i for i in index_list if i not in del_indexs]:
                if self.agents[indx].age == 0:
                    del_indexs.append(indx)
                    amt_death += 1
                    if amt_death > num_death:
                        break

        del_indexs = - np.sort(-np.unique(del_indexs))
        del_indexs = del_indexs[:-1]

        temp_agents = deepcopy(self.agents)
        for indx in range(len(del_indexs)):
            del temp_agents[del_indexs[indx]]
        self.agents = deepcopy(temp_agents)

    def change_culture_base(self):
        for i in range(0, len(self.culture_bases)):
            new_base = np.zeros(self.n)
            indexes = []
            for j in range(0, len(self.agents)):
                if self.cultures[i] == self.agents[j].culture:
                    indexes.append(j)
                    new_base += self.agents[j].culture_state
            new_base = new_base / np.linalg.norm(new_base)

            self.culture_bases[i] = new_base

            angles = [self.angle_vec(new_base, self.agents[k].culture_state) for k in indexes]
            self.critical_angles[i] = np.std(angles) ** 2

    def do_education_process(self, list_clusters):
        mean_culture_state = np.zeros(self.n)
        for cluster in list_clusters:
            for indx_agent in cluster:
                mean_culture_state += self.agents[indx_agent].culture_state
            mean_culture_state /= len(cluster)

            for indx_agent in cluster:
                if self.agents[indx_agent].age == 1:
                    cult = self.agents[indx_agent].culture
                    new_vec = np.ones(self.n)
                    while self.angle_vec(mean_culture_state, new_vec) + \
                            self.education[cult] < self.angle_vec(mean_culture_state, self.agents[indx_agent].culture_state):
                        new_vec = np.random.normal(0, 5, self.n)
                        new_vec /= np.linalg.norm(new_vec)
                    self.agents[indx_agent].culture_state = self.agents[indx_agent].culture_state + self.education[cult] * new_vec
                    self.agents[indx_agent].culture_state /= np.linalg.norm(self.agents[indx_agent].culture_state)

    def power_iteration(self):

        self.graph_list_num_cluster_unique_culture = [0] * len(self.cultures)
        self.graph_num_cluster_cross_culture = 0
        self.graph_list_std_per_culture = []
        self.graph_list_mean_per_culture = []

        for obj in self.agents:
            np.append(obj.culture_memory, obj.culture_state)
        critical_angle = 0.1
        list_clusters = self.clusterization(self.agents, critical_angle)
        pairs, mean_cultures_state = self.clusters_factorization(list_clusters)
        interaction_angle = 0.4

        list_angles_cluster = np.zeros((len(self.agents), 2))
        for pair in pairs:
            state_1 = mean_cultures_state[pair[0]]
            state_2 = mean_cultures_state[pair[1]]

            new_1 = state_1 + np.cos(interaction_angle + self.angle_vec(state_1, state_2)) * np.random.uniform(0, 1) * (state_2 - state_1)
            new_2 = state_2 + np.cos(interaction_angle + self.angle_vec(state_1, state_2)) * np.random.uniform(0, 1) * (state_1 - state_2)

            temp_angle = 0
            for indx, indx_obj in enumerate(list_clusters[pair[0]]):
                list_angles_cluster[indx_obj, 0] = self.agents[indx_obj].culture
                temp_vec = deepcopy(self.agents[indx_obj].culture_state)
                self.agents[indx_obj].culture_state = self.agents[indx_obj].culture_state + self.step * (self.agents[indx_obj].culture_state + new_1) / \
                                                      np.linalg.norm(self.agents[indx_obj].culture_state)
                list_angles_cluster[indx_obj, 1] = self.angle_vec(self.agents[indx_obj].culture_state, temp_vec)

            for indx, indx_obj in enumerate(list_clusters[pair[1]]):
                list_angles_cluster[indx_obj, 0] = self.agents[indx_obj].culture
                temp_vec = deepcopy(self.agents[indx_obj].culture_state)
                self.agents[indx_obj].culture_state = self.agents[indx_obj].culture_state + self.step * (self.agents[indx_obj].culture_state + new_2) / \
                                                      np.linalg.norm(self.agents[indx_obj].culture_state)
                list_angles_cluster[indx_obj, 1] = self.angle_vec(self.agents[indx_obj].culture_state, temp_vec)

        for cult in self.cultures:
            temp_vec = []
            for i in range(len(self.agents)):
                if list_angles_cluster[i, 0] == cult:
                    temp_vec.append(list_angles_cluster[i, 1])

            self.graph_list_std_per_culture.append(np.std(temp_vec))
            self.graph_list_mean_per_culture.append(np.mean(temp_vec))

        for indx, cluster in enumerate(list_clusters):
            temp_vec = []
            for indx_obj in cluster:
                temp_vec.append(self.agents[indx_obj].culture)
            if np.unique(temp_vec).shape[0] == 1:
                self.graph_list_num_cluster_unique_culture[temp_vec[0]] += 1
            else:
                self.graph_num_cluster_cross_culture += 1

        self.culture_rechanging()
        self.do_demography_iteration()

        critical_angle = 0.1
        list_clusters = self.clusterization(self.agents, critical_angle)
        self.do_education_process(list_clusters)
        self.change_culture_base()

        for agent in self.agents:
            agent.age = 0

    @staticmethod
    def angle(obj1, obj2):
        return np.arccos(np.dot(obj1.culture_state, obj2.culture_state) / (np.linalg.norm(obj1.culture_state)
                                                                           * np.linalg.norm(obj2.culture_state)))

    @staticmethod
    def angle_vec(vec1, vec2):
        return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
