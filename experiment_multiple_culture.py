from model.engine import MainEngine
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import torch
import collections


n_steps = 500
size = 100
threshold = 0
n_elements = 0
energy = 0.5
engine = MainEngine(n_elements=n_elements, size=size, threshold=threshold, death=0.00125, birth=0.0019)

vec1 = np.random.normal(0, 0.3, size) + [0 if x % 2 == 0 and x < size/2 else 1 for x in range(size)]
vec2 = np.random.normal(0, 0.3, size) + [1 if x % 2 == 0 and x < size/2 else 0 for x in range(size)]
vec3 = np.random.normal(0, 0.3, size) + [1 if x % 2 == 0 and x > size/2 else 0 for x in range(size)]
vec4 = np.random.normal(0, 0.3, size) + [0 if x % 2 == 0 and x > size/2 else 1 for x in range(size)]

engine.scenario(list_amt=[1500, 2000, 700, 4000], list_cult=[vec1, vec2, vec3, vec4], list_class=[0, 1, 2, 3],
                list_education=[0.5, 0.5, 0.5, 0.5],
                list_fertility=[1, 1, 1, 1], depth_memory=0)

for x in engine.list_obj:
    x.age = np.random.randint(0, 60)
all_amt = [len(engine.list_obj)]
weigths = []

date_now = str(datetime.datetime.now())[:19].replace(':', '_')
date_now = date_now.replace('-', '_')
date_now = date_now.replace(' ', '_')

p = Path(f'./images/experiment_{date_now}/')
p.mkdir()

for i in range(n_steps):
    print(i)
    engine.step(indx=i, energy=energy)

    all_amt.append(len(engine.list_obj))
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    axes[0].plot(all_amt)
    axes[0].set_title('Population')
    axes[0].set_ylabel('#')
    axes[0].set_xlabel('Steps')

    hist_data = [x.sclass for x in engine.list_obj]
    axes[1].hist(hist_data)
    axes[1].set_title('Number of people per class hist')
    axes[1].set_ylabel('#')
    axes[1].set_xlabel('Steps')

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
        plt.savefig(f'./images/experiment_{date_now}/{i}_graph.png')
        torch.save(engine.interaction_model.state_dict(), f'./images/experiment_{date_now}/checkpoint_{i}.pth')
        engine.interaction_model.load_state_dict(torch.load(f'./images/experiment_{date_now}/checkpoint_{i}.pth'))
    plt.savefig(f'./images/experiment_{date_now}/last_graph.png')

    torch.save(engine.interaction_model.state_dict(), f'./images/experiment_{date_now}/checkpoint_last.pth')

print('Done!')