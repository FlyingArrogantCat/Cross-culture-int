from model.engine import MainEngine
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path


n_steps = 500
size = 100
threshold = 0
n_elements = 50
energy = 0.5
engine = MainEngine(n_elements=n_elements, size=size, threshold=threshold)

vec1 = np.random.normal(0, 1, size)
vec2 = np.random.normal(0, 1, size)
engine.scenario(list_amt=[3000, 7000], list_cult=[vec1, vec2], depth_memory=100)
all_amt = [len(engine.list_obj)]
len_class0 = [3000]
len_class1 = [7000]
len_class0w = [0.3]
len_class1w = [0.7]
mean_norm = [np.mean([np.linalg.norm(vec1), np.linalg.norm(vec2)])]

date_now = str(datetime.datetime.now())[:19].replace(':', '_')
date_now = date_now.replace('-', '_')
date_now = date_now.replace(' ', '_')

p = Path(f'./images/experiment_' + date_now + '/')
p.mkdir()

for i in range(n_steps):
    print(i)
    if i == 0:
        engine.step(energy=energy, update=100)
    else:
        engine.step(energy=energy)

    all_amt.append(len(engine.list_obj))
    len_class0.append(len([x for x in engine.list_obj if x.sclass == 0]))
    len_class1.append(len([x for x in engine.list_obj if x.sclass == 1]))
    len_class0w.append(len([x for x in engine.list_obj if x.sclass == 0])/len(engine.list_obj))
    len_class1w.append(len([x for x in engine.list_obj if x.sclass == 1])/len(engine.list_obj))
    mean_norm.append(np.mean([np.linalg.norm(x.culture_condition.detach().numpy())
                              for x in engine.list_obj if x.condition =='torch']))
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))

    axes[0].plot(all_amt)
    axes[0].set_title('Number of people')
    axes[0].set_ylabel('#')
    axes[0].set_xlabel('Steps')

    axes[1].plot(len_class1, label='class 1')
    axes[1].plot(len_class0, label='class 0')
    axes[1].set_title('Number of people per class')
    axes[1].set_ylabel('#')
    axes[1].set_xlabel('Steps')
    axes[1].legend()

    axes[2].plot(mean_norm)
    axes[2].set_title('mean Norm')
    axes[2].set_ylabel('#')
    axes[2].set_xlabel('Steps')

    axes[3].plot(len_class1w, label='class 1w')
    axes[3].plot(len_class0w, label='class 0w')
    axes[3].set_title('weights number of people ')
    axes[3].set_ylabel('#')
    axes[3].set_xlabel('Steps')
    axes[3].legend()

    if i % 20 == 0:
        plt.savefig(f'./images/experiment_{date_now}/{i}_graph.png')
    plt.savefig(f'./images/experiment_{date_now}/last_graph.png')


print('Done!')