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
# beta = 0.1

qcopa_perf = np.zeros(10)
optimum_perf = np.zeros(10)
greedy_perf = np.zeros(10)
simultaneous_perf = np.zeros(10)

#Main loop
cnt = 0;
for bb in tqdm(np.linspace(0,1,10)):

    beta = bb
    actions_1 = np.linspace(Pmin, Pmax_1, Npower)
    actions_2 = np.linspace(Pmin, Pmax_2, Npower)
    states = np.array([0])

    agents = []
    PA_1 = Agent(actions_1.size, actions_2.size)
    PA_2 = Agent(actions_1.size, actions_2.size)
    agents.append(PA_1)
    agents.append(PA_2)

    # Q-learning
    Iterations = 30 * (actions_1.size * actions_2.size)

    for episode in np.arange(Iterations):

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

    # calc reward
    # A_1
    signal = PA_1.power * g1
    interf = PA_2.power * g1 * beta
    reward_1 = R_2(signal, interf, 1.0)

    # A_2
    signal = PA_2.power * g2
    interf = PA_1.power * g2 * beta
    reward_2 = R_2(signal, interf, 1.0)
    qcopa_perf[cnt] = reward_1 + reward_2

    val_simul = np.log2(1 + ((Pmax_1 * g1) / (Pmax_2 * g1 * beta + 1))) + np.log2(1 + (Pmax_2 * g2) / (Pmax_1 * g2 * beta + 1))
    val_0 = np.log2(1 + (Pmax_1 * g1) / ((1) ))
    val_greedy = np.log2(1 + (Pmax_2 * g2) / ((1)))
    val_optimum = max(val_0, val_greedy, val_simul)

    optimum_perf[cnt] = val_optimum
    greedy_perf[cnt] = val_greedy
    simultaneous_perf[cnt] = val_simul
    cnt = cnt+1

t = np.linspace(0,1,10)

# red dashes, blue squares and green triangles
# plt.plot(t, qcopa_perf, 'r--', t, optimum_perf, 'bs', t, greedy_perf, 'g^', t, simultaneous_perf, 'k*')

fig = plt.figure()
l_qcopa, l_opt, l_greedy, l_simul = plt.plot(t, qcopa_perf, t, optimum_perf, t, greedy_perf, t, simultaneous_perf)
# lines = plt.plot(t, qcopa_perf, t, optimum_perf, t, greedy_perf, t, simultaneous_perf)
# or MATLAB style string value pairs
plt.setp(l_qcopa, 'ls', '--' ,'marker', '*', 'color', 'r', 'LineWidth',1,'MarkerSize',15, 'label' , 'Q-CoPA')
plt.setp(l_opt, 'ls', '--','color','k', 'LineWidth',3,'MarkerSize',10, 'label' , 'Optimum')
plt.setp(l_greedy, 'ls', '--', 'marker', 'o', 'color', 'g', 'LineWidth',1,'MarkerSize',5, 'label' , 'Greedy')
plt.setp(l_simul, 'ls', '--', 'marker', 's', 'color', 'b', 'LineWidth',1,'MarkerSize',10, 'label' , 'Simultaneous')

plt.xlabel('Portion of Interference' r'($\beta$)',fontsize=20)
plt.ylabel('Normalized throughput($bps/Hz$)',fontsize=20)
# plt.title('Normalized throughput versus $beta$.',fontsize=20)
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.legend()
plt.show()

fig.savefig('compare.jpg')


