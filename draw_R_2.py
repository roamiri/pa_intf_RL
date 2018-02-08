
# coding: utf-8

# In[111]:


from math import *
import numpy as np
import random
from Agent import Agent
from Rewards import *
import matplotlib.pyplot as plt


# In[112]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# In[113]:

p1_max = 10
p2_max = 20
# Make data.
p1 = np.linspace(0,p1_max, 100)
p2 = np.linspace(0,p2_max, 100)
p1,p2 =np.meshgrid(p1,p2)


# In[114]:


def deriveRate(p1, p2, b1, b2):
    #Channel conditions
    g1 = 2.5
    g2 = 1.5
    sigma2 = 1
    beta1 = b1
    beta2 = b2
    Gamma = 3.532
    R1 = np.log10(1+(g1*p1)/((g1*p2*beta2+sigma2)*Gamma))
    R2 = np.log10(1+(g2*p2)/((g2*p1*beta1+sigma2)*Gamma))
    R = R1+R2
    return R


# In[115]:


def colorMesh(b1, b2):
    colors = []
    for y in range(len(p2)):
        for x in range(len(p1)):
            colors.append([0,0,0])
    return colors
    
# In[116]:


def DrawRate(x, y, z, colors, ax):
    surf = ax.plot_surface(x, y, z, facecolor=colors, linewidth=0)


fig = plt.figure()
ax = fig.gca(projection='3d')
beta1 = 0.1
for beta2 in np.linspace(0.1,0.2,2):
    colors = []
    R = deriveRate(p1, p2, beta1, beta2)
    colors = colorMesh(beta1, beta2)
    DrawRate(p1, p2, R, colors, ax)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

