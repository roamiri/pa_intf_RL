
# coding: utf-8

# In[1]:


from math import *
import numpy as np
import random
from Agent import Agent
from Rewards import *
import matplotlib.pyplot as plt


# In[11]:


Pmin = 0.0
Pmax_1 = 10.0
Pmax_2 = 100.0
Npower = 100
sigma2 = 1

actions_1 = np.linspace(Pmin, Pmax_1, Npower)
actions_2 = np.linspace(Pmin, Pmax_2, Npower)

states = np.array([0])


# In[12]:


alpha = 0.5
gamma = 0.9
epsilon = 0.1
# QSize = actions.size * states.size
# half_size = (int) (0.5*QSize)
epsilon = 0.1*100


# In[13]:


agents = []
PA_1 = Agent(states.size, actions_1.size)
PA_2 = Agent(states.size, actions_2.size)
agents.append(PA_1)
agents.append(PA_2)


# In[14]:


#Channel conditions
g1 = 2.5
g2 = 1.5
Gamma = 3.532
sigma2 = 1
beta = 0.1
optimal = np.log2(1 + ((Pmax_1*g1)/((Pmax_2*g1*beta+1)*Gamma)))+np.log2(1+(Pmax_2*g2)/((Pmax_1*g2*beta+1)*Gamma))
optimal_1 = np.log2(1+(Pmax_1*g1)/((1)*Gamma))
optimal_2 = np.log2(1+(Pmax_2*g2)/((1)*Gamma))


# In[15]:


Iterations = 50*(actions_1.size*actions_2.size)
system_perf = np.zeros((1,Iterations))
Q_joint_1 = np.zeros((actions_1.size,actions_2.size))
Q_joint_2 = np.zeros((actions_1.size,actions_2.size))


# In[16]:



for episode in np.arange(Iterations):

    # Choosing action
    sumQ = np.zeros((states.size, actions_1.size))
    for i in [0, 1]:
        PA = agents[i]
        sumQ += PA.Q

    powers = np.array([0,0])

    for i in [0, 1]:
        if i==0:
            actions = actions_1
        else:
            actions = actions_2
        PA = agents[i]
        if (episode/Iterations*100) < 80:
            rnd = random.randint(1,100)
            if rnd< epsilon:
                idx = random.randint(0,Npower-1)
                PA.set_power(actions[idx])
                PA.p_index = idx
            else:
                max_indice = np.argmax(PA.Q, axis = 1)
                idx = max_indice[PA.s_index]
                PA.p_index = idx
                PA.set_power(actions[idx])
        else:
            max_indice = np.argmax(PA.Q, axis=1)
            idx = max_indice[PA.s_index]
            PA.p_index = idx
            PA.set_power(actions[idx])
        powers[i] = PA.power
        agents[i] = PA

    # Calculate the Reward
    # Agent1
    PA_1 = agents[0]
    PA_2 = agents[1]
    
    signal = PA_1.power*g1
    interf = PA_2.power*g1*beta
    reward_1 = R_2(signal, interf, 1.0)
    reward_1_joint = (signal-interf)
    
    act = PA_1.p_index
    st = PA_1.s_index
    PA_1.Q[st, act] = PA_1.Q[st, act] + alpha * (reward_1 - PA_1.Q[st, act])

    # Agent2
    signal = PA_2.power * g2
    interf = PA_1.power * g2 * beta
    reward_2 = R_2(signal, interf, 1.0)
    reward_2_joint = (signal-interf)
    act = PA_2.p_index
    st = PA_2.s_index
    PA_2.Q[st, act] = PA_2.Q[st, act] + alpha * (reward_2 - PA_2.Q[st, act])

    system_perf[0,episode] = reward_1 + reward_2
    ii = PA_1.p_index
    jj = PA_2.p_index
    Q_joint_1[ii, jj] = Q_joint_1[ii, jj] + alpha * ( reward_1_joint - Q_joint_1[ii, jj])
    Q_joint_2[ii, jj] = Q_joint_2[ii, jj] + alpha * ( reward_2_joint - Q_joint_2[ii, jj])


# In[17]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(1)
ax = fig.gca(projection='3d')
        
# Plot the surface.
X , Y = np.meshgrid(actions_2, actions_1)
surf_1 = ax.plot_surface(X, Y, Q_joint_1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.matshow(Q_joint)
plt.show()

fig_2 = plt.figure(2)
ax2 = fig_2.gca(projection='3d')
X, Y = np.meshgrid(actions_2, actions_1)
surf_2 = ax2.plot_surface(X, Y, Q_joint_2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
# input()


