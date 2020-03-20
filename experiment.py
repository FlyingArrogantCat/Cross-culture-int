from model.model_new_geometry.engine import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import collections


name_experiment = 'geom'
k = 1
m = 2
mu = 0.3
angles = [0.4, 0.4, 0.4]
cultures = [0, 1, 2]
amt_member = [100, 50, 20]
bases = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
crit_angle = 0.05
n_steps = 200

engine = MainEngine(k=k, m=m)  #death=0.013, birth=0.0115
engine.define_demography(scale_b=0.17, scale_d=0.15, death_iter_border=100)
engine.initialize_experiment(cultures, amt_member, bases, critical_angles=angles,
                             education_scales=[0.1, 0.1, 0.1], mu=mu)

date_now = str(datetime.datetime.now())[:19].replace(':', '_')
date_now = date_now.replace('-', '_')
date_now = date_now.replace(' ', '_')

p = Path(f'./images/{name_experiment}_experiment_{date_now}/')
p.mkdir()

hist_num = []
cult1 = []
cult2 = []
cult3 = []
list_std_per_cult = []
list_mean_per_cult = []
list_num_cluster_per_cult = []
list_cross_cult = []

for i in range(n_steps):
    print(i)
    engine.power_iteration()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    hist_num.append(len(engine.agents))
    axes[0, 0].plot(hist_num)
    axes[0, 0].set_title('Population')
    axes[0, 0].set_ylabel('#')
    axes[0, 0].set_xlabel('Steps')

    cult1.append(len([x.culture for x in engine.agents if x.culture == 0]))
    cult2.append(len([x.culture for x in engine.agents if x.culture == 1]))
    cult3.append(len([x.culture for x in engine.agents if x.culture == 2]))
    axes[0, 1].plot(cult1, label='class 0')
    axes[0, 1].plot(cult2, label='class 1')
    axes[0, 1].plot(cult3, label='class 2')
    axes[0, 1].set_title('Number of people per class')
    axes[0, 1].set_ylabel('#')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].legend()

    list_std_per_cult.append(engine.graph_list_std_per_culture)
    list_mean_per_cult.append(engine.graph_list_mean_per_culture)

    axes[1, 0].plot(np.array(list_std_per_cult)[:, 0], label='class 0')
    axes[1, 0].plot(np.array(list_std_per_cult)[:, 1], label='class 1')
    axes[1, 0].plot(np.array(list_std_per_cult)[:, 2], label='class 2')
    axes[1, 0].set_title('Standard deviation')
    axes[1, 0].set_ylabel('#')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].legend()

    list_cross_cult.append(engine.graph_num_cluster_cross_culture)
    list_num_cluster_per_cult.append(engine.graph_list_num_cluster_unique_culture)

    axes[1, 1].plot(np.array(list_num_cluster_per_cult)[:, 0], label='class 0')
    axes[1, 1].plot(np.array(list_num_cluster_per_cult)[:, 1], label='class 1')
    axes[1, 1].plot(np.array(list_num_cluster_per_cult)[:, 2], label='class 2')
    axes[1, 1].plot(list_cross_cult, label='cross_cult_cluster')
    axes[1, 1].set_title('Clusters information')
    axes[1, 1].set_ylabel('#')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].legend()

    if i % 20 == 0:
        plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/{i}_graph.png')
    plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/last_graph.png')
    plt.close(fig)

print('Done!')
