from model.model_new_geometry.engine import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from pathlib import Path
import pandas as pd
import json
from copy import deepcopy
import shutil
import sys


def main():
    name_experiment = 'sep_exp'

    k = 1
    m = 2
    mu = 0.3
    angles = [0.5, 0.5, 0.5]
    cultures = [0, 1, 2]
    amt_member = [100, 50, 20]
    bases = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    fertility = [1, 1, 1]
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
    img_path = p / 'images'
    img_path.mkdir()

    path_curr_script = Path.cwd()/sys.argv[0]
    shutil.copy(str(path_curr_script), str(p))

    internal_parameters = {'k': k, 'm': m, 'mu': mu, 'cult_angles': angles, 'amt_member': amt_member, 'bases': bases,
                           'fertility': fertility, 'cricical_anlge_cluster': crit_angle, 'education': education}
    json_file = {'date_experiment': str(datetime.datetime.now())[:19], 'internal_param': internal_parameters}
    with open(f'./images/{name_experiment}_experiment_{date_now}/params.json', 'w') as outfile:
        json.dump(json_file, outfile)

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
        engine.power_iteration(cult_appear=False)

        fig1, ax1 = plt.subplots(figsize=(15, 15))
        fig2, ax2 = plt.subplots(figsize=(15, 15))
        fig3, ax3 = plt.subplots(figsize=(15, 15))
        fig4, ax4 = plt.subplots(figsize=(15, 15))

        matplotlib.rcParams.update({'font.size': 15})

        hist_num.append(len(engine.agents))
        ax1.plot(hist_num)

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
            ax2.plot(list_cult_amt[culture], linestyle=linestyles[culture], label=name_str)
            ax3.plot(np.array(list_std_per_cult)[:, culture], linestyle=linestyles[culture], label=name_str)
            ax4.plot(np.array(list_num_cluster_per_cult)[:, culture], linestyle=linestyles[culture], label=name_str)

        ax1.set_title('Численность')
        ax1.set_ylabel('#')
        ax1.set_xlabel('Итерация')

        ax2.set_title('Численность по культурам')
        ax2.set_ylabel('#')
        ax2.set_xlabel('Итерация')
        ax2.legend()

        ax3.set_title('Стандартное отклонение $\sigma$')
        ax3.set_ylabel('#')
        ax3.set_xlabel('Итерация')
        ax3.legend()

        ax4.plot(list_cross_cult, label='межкультурные кластеры')
        ax4.set_title('Число кластеров по культурам')
        ax4.set_ylabel('#')
        ax4.set_xlabel('Итерации')
        ax4.legend()

        if i % 20 == 0:
            fig1.savefig(str(img_path/ f'{i}_general_amt.png'))
            fig2.savefig(str(img_path/ f'{i}_culture_amt.png'))
            fig3.savefig(str(img_path/ f'{i}_std.png'))
            fig4.savefig(str(img_path/ f'{i}_clusters.png'))

        fig1.savefig(str(img_path / f'last_general_amt.png'))
        fig2.savefig(str(img_path / f'last_culture_amt.png'))
        fig3.savefig(str(img_path / f'last_std.png'))
        fig4.savefig(str(img_path / f'last_clusters.png'))

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

        df_dict = {'hist_num': hist_num,'cluster_cross_cult': list_cross_cult}

        for culture in temp_cultures:
            df_dict['cult_amt_'+str(culture)] = list_cult_amt[culture]
            df_dict['std_cult_'+str(culture)] = np.array(list_std_per_cult)[:, culture]
            df_dict['cluster_cult_'+str(culture)] = np.array(list_num_cluster_per_cult)[:, culture]

        df = pd.DataFrame(df_dict)
        df.to_csv(str(p/'DataFrame_last.csv'))

    print('Done!')


if __name__ == '__main__':
    main()
