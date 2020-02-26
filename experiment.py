from model.model_new_geometry.engine import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import collections


name_experiment = 'new_test_geom'
k = 1
m = 1
mu = 0.3
angles = [0.3, 0.3]
cultures = [0, 1]
amt_member = [50, 50]
bases = [[0, 1], [1, 0]]
crit_angle = 0.05
n_steps = 200

engine = MainEngine(k=k, m=m)  #death=0.013, birth=0.0115
engine.define_demography(scale_b=0.15, scale_d=0.0515, death_iter_border=100)
engine.initialize_experiment(cultures, amt_member, bases, critical_angles=angles, mu=mu)

date_now = str(datetime.datetime.now())[:19].replace(':', '_')
date_now = date_now.replace('-', '_')
date_now = date_now.replace(' ', '_')

p = Path(f'./images/{name_experiment}_experiment_{date_now}/')
p.mkdir()

hist_num = []
cult1 = []
cult2 = []

for i in range(n_steps):
    print(i)
    engine.power_iteration()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    hist_num.append(len(engine.agents))
    axes[0].plot(hist_num)
    axes[0].set_title('Population')
    axes[0].set_ylabel('#')
    axes[0].set_xlabel('Steps')

    cult1.append(len([x.culture for x in engine.agents if x.culture == 0]))
    cult2.append(len([x.culture for x in engine.agents if x.culture == 1]))

    axes[1].plot(cult1, label='class 0')
    axes[1].plot(cult2, label='class 1')
    axes[1].set_title('Number of people per class')
    axes[1].set_ylabel('#')
    axes[1].set_xlabel('Steps')
    axes[1].legend()

    if i % 20 == 0:
        plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/{i}_graph.png')
    plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/last_graph.png')
    plt.close(fig)

print('Done!')
