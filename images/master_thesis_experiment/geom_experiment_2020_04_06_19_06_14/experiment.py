from model.model_new_geometry.engine import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from pathlib import Path
import collections
from copy import deepcopy
import shutil
import sys


def main():
    name_experiment = 'geom'
    k = 1
    m = 2
    mu = 0.3
    angles = [0.5, 0.5, 0.5]
    cultures = [0, 1, 2]
    amt_member = [100, 50, 20]
    bases = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    fertility = [1, 1.1, 1.1]
    crit_angle = 0.05
    n_steps = 200
    education = [0.1, 0.1, 0.1]

    engine = MainEngine(k=k, m=m)
    engine.define_demography(scale_b=0.15, scale_d=0.15)
    engine.initialize_experiment(cultures, amt_member, bases, critical_angles=angles,
                                 education_scales=education, mu=mu, fertility=fertility)

    date_now = str(datetime.datetime.now())[:19].replace(':', '_')
    date_now = date_now.replace('-', '_')
    date_now = date_now.replace(' ', '_')

    p = Path(f'./images/{name_experiment}_experiment_{date_now}/')
    p.mkdir()

    path_curr_script = Path.cwd()/sys.argv[0]
    shutil.copy(str(path_curr_script), str(p))

    hist_num = []
    list_cult_amt = []
    for i in range(0, len(cultures)):
        list_cult_amt.append([])

    list_std_per_cult = []
    list_mean_per_cult = []
    list_num_cluster_per_cult = []
    list_cross_cult = []

    linestyles = ['-', '--', '-.', 'dotted', 'dashdot', ':', 'solid']
    flag_new_cult = 0
    for i in range(n_steps):
        temp_cultures = deepcopy(engine.cultures)
        print(i)
        engine.power_iteration()

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        matplotlib.rcParams.update({'font.size': 14})

        hist_num.append(len(engine.agents))
        axes[0, 0].plot(hist_num)
        axes[0, 0].set_title('Численность')
        axes[0, 0].set_ylabel('#')
        axes[0, 0].set_xlabel('Итерация')

        print('new_cultures:', engine.new_cultures)
        if int(engine.new_cultures) == 1 and flag_new_cult == 0 and len(temp_cultures) == 4:
            print('NEW CULTURE')
            list_cult_amt.append([0] * len(list_cult_amt[0]))
            flag_new_cult += 1

            for indx in range(len(list_std_per_cult)):
                if len(list_std_per_cult[indx]) < len(temp_cultures):
                    list_std_per_cult[indx].append(0)
                    list_num_cluster_per_cult[indx].append(0)
            print('End change matrixes')

        list_cross_cult.append(engine.graph_num_cluster_cross_culture)
        list_num_cluster_per_cult.append(engine.graph_list_num_cluster_unique_culture)
        list_std_per_cult.append(engine.graph_list_std_per_culture)
        list_mean_per_cult.append(engine.graph_list_mean_per_culture)

        print('len_cult:', len(temp_cultures))
        for culture in temp_cultures:

            list_cult_amt[culture].append(len([x.culture for x in engine.agents if x.culture == culture]))

            name_str = f'культура {culture}'
            axes[0, 1].plot(list_cult_amt[culture], linestyle=linestyles[culture], label=name_str)
            axes[1, 0].plot(np.array(list_std_per_cult)[:, culture], linestyle=linestyles[culture], label=name_str)
            axes[1, 1].plot(np.array(list_num_cluster_per_cult)[:, culture], linestyle=linestyles[culture], label=name_str)

        axes[0, 1].set_title('Численность по культурам')
        axes[0, 1].set_ylabel('#')
        axes[0, 1].set_xlabel('Итерация')
        axes[0, 1].legend()

        axes[1, 0].set_title('Стандартное отклонение $\sigma$')
        axes[1, 0].set_ylabel('#')
        axes[1, 0].set_xlabel('Итерация')
        axes[1, 0].legend()

        axes[1, 1].plot(list_cross_cult, label='межкультурные кластеры')
        axes[1, 1].set_title('Число кластеров по культурам')
        axes[1, 1].set_ylabel('#')
        axes[1, 1].set_xlabel('Итерации')
        axes[1, 1].legend()

        if i % 20 == 0:
            plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/{i}_graph.png')
        plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/last_graph.png')
        plt.close(fig)

    print('Done!')


if __name__ == '__main__':
    main()
