from model.model_with_nn.engine import MainEngine
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import torch
import collections


name_experiment = 'test'
n_steps = 500
size = 100
threshold = 0
n_elements = 0
energy = 500
engine = MainEngine(n_elements=n_elements, size=size, threshold=threshold, death=0, birth=0)#death=0.013, birth=0.0115)

vec1 = np.random.normal(0, 0.3, size) + [10 if x < size/3 else 0 for x in range(size)]
vec2 = np.random.normal(0, 0.3, size) + [10 if size/3 < x < 2 * size/3 else 0 for x in range(size)]
vec3 = np.random.normal(0, 0.3, size) + [10 if x > 2 * size/3 else 0 for x in range(size)]


engine.scenario(list_amt=[3000, 1500, 500], list_cult=[vec1, vec2, vec3], list_class=[0, 1, 2],
                list_education=[0.5, 0.5, 0.5],
                list_fertility=[1, 1, 1], depth_memory=10, give_mem_child=False)

for x in engine.list_obj:
    x.age = np.random.randint(0, 60)
all_amt = [len(engine.list_obj)]
weigths = []
class0_data = []
class1_data = []
class2_data = []
date_now = str(datetime.datetime.now())[:19].replace(':', '_')
date_now = date_now.replace('-', '_')
date_now = date_now.replace(' ', '_')

p = Path(f'./images/{name_experiment}_experiment_{date_now}/')
p.mkdir()


#engine.interaction_model.load_state_dict(torch.load('/home/fedor/projects/Master-project/checkpoint.pth'))

for i in range(n_steps):
    print(i)
    if i== 0:
        engine.step(indx=i, constant=500, energy=energy)
    else:
        engine.step(indx=i, constant=500, energy=energy)

    all_amt.append(len(engine.list_obj))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(all_amt)
    axes[0].set_title('Population')
    axes[0].set_ylabel('#')
    axes[0].set_xlabel('Steps')

    class0_data.append(len([x.sclass for x in engine.list_obj if x.sclass == 0]))
    class1_data.append(len([x.sclass for x in engine.list_obj if x.sclass == 1]))
    class2_data.append(len([x.sclass for x in engine.list_obj if x.sclass == 2]))
    hist_data = [x.sclass for x in engine.list_obj]
    axes[1].plot(class0_data, label='class 0')
    axes[1].plot(class1_data, label='class 1')
    axes[1].plot(class2_data, label='class 2')
    axes[1].set_title('Number of people per class')
    axes[1].set_ylabel('#')
    axes[1].set_xlabel('Steps')
    axes[1].legend()
    counter = collections.Counter()
    for x in hist_data:
        counter[x] += 1
    counter = dict(counter)
    weigths.append([x / len(hist_data) for x in counter.values()])
    print(weigths[-1])
    for sclass in engine.list_class:
        temp_weigths = np.array(weigths)

        axes[2].plot(temp_weigths[:, sclass], label=f'class {sclass}w')
    axes[2].set_title('weights number of people ')
    axes[2].set_ylabel('#')
    axes[2].set_xlabel('Steps')
    axes[2].legend()

    if i % 20 == 0:
        plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/{i}_graph.png')
        #torch.save(engine.interaction_model.state_dict(), f'./images/{name_experiment}_experiment_{date_now}/checkpoint_{i}.pth')
        #engine.interaction_model.load_state_dict(torch.load(f'./images/{name_experiment}_experiment_{date_now}/checkpoint_{i}.pth'))
    plt.savefig(f'./images/{name_experiment}_experiment_{date_now}/last_graph.png')
    plt.close(fig)
    torch.save(engine.interaction_model.state_dict(), f'./images/{name_experiment}_experiment_{date_now}/checkpoint_last.pth')

print('Done!')