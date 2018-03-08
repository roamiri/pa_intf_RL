from math import *
import numpy as np
import random
from Agent import Agent
from Rewards import *
import matplotlib.pyplot as plt


# Parameters
Pmin = 0.0
Pmax_1 = 10.0
Pmax_2 = 100.0
Npower = 100
sigma2 = 1
alpha = 0.5
gamma = 0.9
epsilon = 0.1
# QSize = actions.size * states.size
# half_size = (int) (0.5*QSize)
epsilon = 0.1*100

#Channel conditions
g1 = 2.5
g2 = 1.5
Gamma = 3.532
sigma2 = 1
beta = 0.2
optimal = np.log2(1 + ((Pmax_1*g1)/((Pmax_2*g1*beta+1)*Gamma)))+np.log2(1+(Pmax_2*g2)/((Pmax_1*g2*beta+1)*Gamma))
optimal_1 = np.log2(1+(Pmax_1*g1)/((1)*Gamma))
optimal_2 = np.log2(1+(Pmax_2*g2)/((1)*Gamma))

actions_1 = np.linspace(Pmin, Pmax_1, Npower)
actions_2 = np.linspace(Pmin, Pmax_2, Npower)
states = np.array([0])

agents = []
PA_1 = Agent(actions_2.size, actions_1.size)
PA_2 = Agent(actions_1.size, actions_2.size)
agents.append(PA_1)
agents.append(PA_2)

#Q-learning
Iterations = 50*(actions_1.size*actions_2.size)
system_perf = np.zeros((1,Iterations))

#Main loop
for episode in np.arange(Iterations):
