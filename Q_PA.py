from math import *
import numpy as np
import random
from Agent import Agent
from Rewards import *
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import time
from tqdm import tqdm

# Parameters
Pmin = 0.0
Pmax_1 = 10.0
Pmax_2 = 20.0
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
beta = 0.1
optimal = np.log2(1 + ((Pmax_1*g1)/((Pmax_2*g1*beta+1)*Gamma)))+np.log2(1+(Pmax_2*g2)/((Pmax_1*g2*beta+1)*Gamma))
optimal_1 = np.log2(1+(Pmax_1*g1)/((1)*Gamma))
optimal_2 = np.log2(1+(Pmax_2*g2)/((1)*Gamma))

actions_1 = np.linspace(Pmin, Pmax_1, Npower)
actions_2 = np.linspace(Pmin, Pmax_2, Npower)
states = np.array([0])

agents = []
PA_1 = Agent(states.size, actions_1.size)
PA_2 = Agent(states.size, actions_2.size)
agents.append(PA_1)
agents.append(PA_2)

#Q-learning
Iterations = 50*(actions_1.size*actions_2.size)
system_perf = np.zeros((1,Iterations))

for episode in tqdm(np.arange(Iterations)):

    # Choosing action
    sumQ = np.zeros((states.size, actions_1.size))
    for i in [0, 1]:
        PA = agents[i]
        sumQ += PA.Q

    powers = np.zeros(2)

    for i in [0, 1]:
        PA = agents[i]
        if i==0:
            actions = actions_1
        else:
            actions = actions_2
        if (episode/Iterations*100) < 80:
            rnd = random.randint(1,100)
            if rnd< epsilon:
                idx = random.randint(0,Npower-1)
                PA.set_power(actions[idx])
                PA.p_index = idx
            else:
                max_indice = np.argmax(sumQ, axis = 1)
                idx = max_indice[PA.s_index]
                PA.p_index = idx
                PA.set_power(actions[idx])
        else:
            max_indice = np.argmax(sumQ, axis=1)
            idx = max_indice[PA.s_index]
            PA.p_index = idx
            PA.set_power(actions[idx])
        powers[i] = PA.power
        agents[i] = PA

    # Calculate the Reward
    # Agent1
    signal = powers[0]*g1
    interf = powers[1]*g1*beta
    reward_1 = R_2(signal, interf, 1.0)
    PA = agents[0]
    act = PA.p_index
    st = PA.s_index
    PA.Q[st, act] = PA.Q[st, act] + alpha * (reward_1 - PA.Q[st, act])

    # Agent2
    signal = powers[1] * g2
    interf = powers[0] * g2 * beta
    reward_2 = R_2(signal, interf, 1.0)
    PA = agents[1]
    act = PA.p_index
    st = PA.s_index
    PA.Q[st, act] = PA.Q[st, act] + alpha * (reward_2 - PA.Q[st, act])

    system_perf[0,episode] = reward_1 + reward_2

fig = plt.figure(1)
ax = fig.gca(projection='3d')

# Plot the surface.
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

Q_total = PA_1.Q + PA_2.Q
X, Y = np.meshgrid(actions_2, actions_1)
surf_1 = ax.plot_surface(X, Y, Q_total, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.spines['bottom'].set_color('0.5')
ax.spines['top'].set_color('0.5')
ax.spines['right'].set_color('0.5')
ax.spines['left'].set_color('0.5')

# plt.title('Global action-value function', fontdict=font)
# plt.text(PA_1.power, PA_2.power, '$\max_{P_1,P2} Q$', fontdict=font)
plt.xlabel('$P_2(mW)$', fontdict=font)
plt.ylabel('$P_1$(mW)', fontdict=font)
#plt.zlabel('$Q(P_1,P_2)$', fontdict=font)

# Add a color bar which maps values to colors.
fig.colorbar(surf_1, shrink=0.5, aspect=10)
# plt.savefig('Q_1.eps', format='eps', dpi=1000)
plt.show()