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
PA_1 = Agent(actions_1.size, actions_2.size)
PA_2 = Agent(actions_1.size, actions_2.size)
agents.append(PA_1)
agents.append(PA_2)

#Q-learning
Iterations = 50*(actions_1.size*actions_2.size)
system_perf = np.zeros((1,Iterations))

#Main loop

for episode in tqdm(np.arange(Iterations)):

    if (episode / Iterations * 100) < 80:
        rnd = random.randint(1, 100)
        if rnd < epsilon:
            idx = random.randint(0, Npower - 1)
            PA_1.set_power(actions_1[idx])
            PA_1.p_index = idx
            idx = random.randint(0, Npower - 1)
            PA_2.set_power(actions_2[idx])
            PA_2.p_index = idx
        else:
            # VE algorithm
            # Pass Q_1 to A_2
            # construct f_2 and B_2
            Q_sum = PA_1.Q + PA_2.Q
            B_2 = np.argmax(Q_sum, axis=1)
            f_2 = np.amax(Q_sum, axis=1)
            # pass f_2 to A_1
            f_1 = max(f_2)
            a_1 = np.argmax(f_2)
            # pass a_1 to A_2
            a_2 = B_2.item(a_1)
            # end of VE

            # Take action by agents
            PA_1.p_index = a_1
            PA_1.set_power(actions_1[a_1])
            PA_2.p_index = a_2
            PA_2.set_power(actions_2[a_2])
    else:
        # VE algorithm
        # Pass Q_1 to A_2
        # construct f_2 and B_2
        Q_sum = PA_1.Q + PA_2.Q
        mm = np.max(Q_sum)
        B_2 = np.argmax(Q_sum, axis=1)
        f_2 = np.amax(Q_sum, axis=1)
        # pass f_2 to A_1
        f_1 = max(f_2)
        a_1 = np.argmax(f_2)
        # pass a_1 to A_2
        a_2 = B_2.item(a_1)
        # end of VE

        # Take action by agents
        PA_1.p_index = a_1
        PA_1.set_power(actions_1[a_1])
        PA_2.p_index = a_2
        PA_2.set_power(actions_2[a_2])



    # calc reward
    #A_1
    signal = PA_1.power * g1
    interf = PA_2.power * g1 * beta
    reward_1 = R_2(signal, interf, 1.0)

    # A_2
    signal = PA_2.power * g2
    interf = PA_1.power * g2 * beta
    reward_2 = R_2(signal, interf, 1.0)

    act1 = PA_1.p_index
    act2 = PA_2.p_index

    PA_1.Q[act1, act2] = PA_1.Q[act1, act2] + alpha * (reward_1 - PA_1.Q[act1, act2])
    PA_2.Q[act1, act2] = PA_2.Q[act1, act2] + alpha * (reward_2 - PA_2.Q[act1, act2])

    system_perf[0, episode] = reward_1 + reward_2

act1 = PA_1.p_index
act2 = PA_2.p_index
print(act1)
print(PA_1.power)
print(act2)
print(PA_2.power)

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
plt.xlabel('$P_2$(mW)', fontdict=font)
plt.ylabel('$P_1$(mW)', fontdict=font)
#plt.zlabel('$Q(P_1,P_2)$', fontdict=font)

# Add a color bar which maps values to colors.
fig.colorbar(surf_1, shrink=0.5, aspect=10)
# plt.savefig('Q_1.eps', format='eps', dpi=1000)
plt.show()


