import torch
from torch import nn
import json
from model.logger import logger
import numpy as np
from model.engine import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class EnergyExperiment(torch.nn.Module):
    def __init__(self, name='TestEnergyExperiment', size_space=100, experiment_path='./experiments/', demography_flag=False,
                 depth_memory=100):
        super(EnergyExperiment, self).__init__()

        if not Path(experiment_path).is_dir():
            Path(experiment_path).mkdir()

        self.main_apth = Path(experiment_path + name)
        if not self.main_apth.is_dir():
            self.main_apth.mkdir()

        self.logger = logger(path=experiment_path + name)
        self.list_energies = None
        self.n_steps = 500

        vec1 = np.random.normal(0, 0.2, size_space) + [1 if x < size_space / 2 else 0 for x in range(size_space)]
        vec2 = np.random.normal(0, 0.3, size_space) + [1 if x >= size_space / 2 else 0 for x in range(size_space)]

        list_amt = [200, 200]

        if demography_flag:
            self.engine = MainEngine(n_elements=0, size=size_space, death=0.013, birth=0.0119)
        else:
            self.engine = MainEngine(n_elements=0, size=size_space, death=0, birth=0)

        print(self.engine.birth)
        print(self.engine.death)

        self.engine.scenario(list_amt=list_amt, list_cult=[vec1, vec2], list_class=[0, 1], depth_memory=depth_memory,
                             list_education=[0.5, 0.5], list_fertility=[1, 1], give_mem_child=True)

        for x in self.engine.list_obj:
            x.age = np.random.randint(0, 60)

    def make_experiment(self, list_energies=None, n_steps=None):

        if n_steps is not None:
            self.n_steps = n_steps

        if list_energies is None:
            self.list_energies = [500]
        else:
            self.list_energies = list_energies

        dataframe = {}
        dataframe['indx'] = range(n_steps)

        for energy in self.list_energies:
            hist_demogr = []
            ratios_class0 = []
            ratios_class1 = []

            path = self.main_apth/str(energy)
            path.mkdir()

            for indx in range(self.n_steps):
                hist_demogr.append(len(self.engine.list_obj))
                ratios_class0.append(len([x.sclass for x in self.engine.list_obj if x.sclass == 0]))
                ratios_class1.append(len([x.sclass for x in self.engine.list_obj if x.sclass == 1]))

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                axes[0].plot(hist_demogr)
                axes[0].set_title('Population')
                axes[0].set_ylabel('#')
                axes[0].set_xlabel('Steps')

                axes[1].plot(ratios_class0, label='class 1')
                axes[1].plot(ratios_class0, label='class 2')
                axes[1].set_title('Number of people per class')
                axes[1].set_ylabel('#')
                axes[1].set_xlabel('Steps')
                axes[1].legend()

                self.engine.step(indx=indx, constant=int(len(self.engine.list_obj) * 0.2),
                                 energy=energy * np.random.normal(1, 0.05))

                if indx % int(self.n_steps/ 10) == 0:
                    plt.savefig(str(path/f'{int(indx)}_graph.png'))

                plt.savefig(str(path/'last_graph.png'))
                plt.close(fig)
                torch.save(self.engine.interaction_model.state_dict(), str(path/'checkpoint_last.pth'))

                self.engine.interaction_model_update()


            dataframe[str(energy) + '/population'] = hist_demogr
            dataframe[str(energy) + '/class0'] = ratios_class0
            dataframe[str(energy) + '/class2'] = ratios_class1

            #with open(str(path/'results.json'), 'w') as f:
            #    json.dump(dataframe, f)

            dataframe = pd.DataFrame(dataframe)
            dataframe.to_csv(str(path / 'results.csv'))
