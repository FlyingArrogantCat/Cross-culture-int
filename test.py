from model.model_new_geometry.engine import *
from model.model_new_geometry.agent import *
from model.model_new_geometry.interaction import *

k = 1
m = 1
mu = 0.3
cultures = [0, 1]
amt_member = [10, 10]
bases = [[0, 1], [1, 0]]
crit_angle = 0.4

engine = MainEngine(k, m)
engine.initialize_experiment(cultures, amt_member, bases, mu)

print('Initialize experiment with parameters: ')
print('k: ', k)
print('m: ', m)
print('cultures: ', cultures)
print('Number of people per culture: ', amt_member)
print('Bases of the culures: ', bases)
print('Mu: ', mu)

print('\nTest:')
print('Num_agents: ', len(engine.agents))
print('Result of the clusterization: ', engine.clusterization(engine.agents, crit_angle))
