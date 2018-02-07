# Power Allocation for a Downlink two-user interference channel
# Sum power Constraints

from math import *
import numpy as np
import random
from Agent import Agent
from R_1 import *
import matplotlib.pyplot as plt

Pmin = 0.0
Pmax = 10.0
Npower = 100
sigma2 = 1

actions = np.linspace(Pmin, Pmax, Npower)

states = np.array([0,1])

alpha = 0.5
gamma_rl = 0.9
epsilon = 0.1
Iterations = 10*actions.size * states.size
epsilon = 0.1*100

#Channel conditions
g1 = 2.5
g2 = 1.5
gamma = 3.532
sigma2 = 1
beta = 0.00
# Main loop

PA = Agent(states.size, actions.size)

total_perf = np.zeros((1, 100))

if g1 >= g2:
    PA.s_index = 0
else:
    PA.s_index = 1

cnt = 0
for beta in np.linspace(0.0, 0.3, 100):
    system_perf = np.zeros((1, Iterations))
    for episode in np.arange(Iterations):

        # Choosing action
        if (episode / Iterations * 100) < 80:
            rnd = random.randint(1, 100)
            if rnd < epsilon:
                idx = random.randint(0, Npower - 1)
                PA.set_power(actions[idx])
                PA.p_index = idx
            else:
                max_indice = np.argmax(PA.Q, axis=1)
                idx = max_indice[PA.s_index]
                PA.p_index = idx
                PA.set_power(actions[idx])
        else:
            max_indice = np.argmax(PA.Q, axis=1)
            idx = max_indice[PA.s_index]
            PA.p_index = idx
            PA.set_power(actions[idx])
        # Calculate the Reward
        p1 = PA.power
        p2 = Pmax - p1
        reward = R_1(p1, p2, g1, g2, beta, sigma2, gamma)
        system_perf[0, episode] = reward
        act = PA.p_index
        st = PA.s_index
        PA.Q[st, act] = PA.Q[st, act] + alpha * (reward - PA.Q[st, act])
    total_perf[0, cnt] = system_perf[0, Iterations - 1]

print('output = ',  PA.power)
plt.plot(system_perf)
plt.show()