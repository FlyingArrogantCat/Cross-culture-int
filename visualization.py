from model.engine import MainEngine
import numpy as np
import matplotlib.pyplot as plt


n_steps = 100
size = 100
threshold = 0
n_elements = 50
energy = 0.5

engine = MainEngine(n_elements=n_elements, size=size, threshold=threshold)
zeros_cult_cond = []
rn_cult_cond = []
energy_hist = []
first = engine.list_obj[0].culture_condition
for i in range(n_steps):
    print(i)
    if i == 0:
        engine.step(energy=energy, update=10)
    else:
        engine.step(energy=energy)
    zeros_cult_cond.append(np.linalg.norm(engine.list_obj[0].culture_condition.detach().numpy(), 2))
    rn_cult_cond.append(np.linalg.norm(engine.list_obj[5].culture_condition.detach().numpy(), 2))
    energy_hist.append(engine.list_obj[0].curr_energy)

    #res = engine.list_obj[0].culture_condition
    #print(np.linalg.norm(first - res.detach().numpy(), 2))
    plt.plot(zeros_cult_cond)
    plt.plot(rn_cult_cond)
    plt.savefig(f'{i}_graph.png')
    #plt.show()
plt.plot(energy_hist)
plt.show()
