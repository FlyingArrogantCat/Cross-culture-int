from model.engine import MainEngine
import numpy as np


n_steps = 2
size = 100
threshold = 0
n_elements = 100
energy = 0.5


engine = MainEngine(n_elements=100, size=size, threshold=threshold)

first = engine.list_obj[0].culture_condition
for i in range(n_steps):
    engine.step(energy=energy)
    res = engine.list_obj[0].culture_condition
    print(np.linalg.norm(first - res.detach().numpy(), 2))
