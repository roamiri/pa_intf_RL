
# coding: utf-8

# In[1]:


from math import *
import numpy as np
import random
from Agent import Agent
from Rewards import *
import matplotlib.pyplot as plt


# In[7]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# In[17]:


#Channel conditions
g1 = 2.5
g2 = 1.5
p1_max = 10
p2_max = 20
sigma2 = 1
beta1 = 0.1
beta2 = 0.5
Gamma = 3.532


# In[27]:


# Make data.
p1 = np.linspace(0,p1_max, 100)
p2 = np.linspace(0,p2_max, 100)
p1,p2 =np.meshgrid(p1,p2)


# In[28]:


R1 = np.log10(1+(g1*p1)/((g1*p2*beta2+sigma2)*Gamma))


# In[29]:


R2 = np.log10(1+(g2*p2)/((g2*p1*beta1+sigma2)*Gamma))


# In[30]:


R3d_1 = R1+R2


# In[39]:

beta1 = 0.1
beta2 = 0.1

R3 = np.log10(1+(g1*p1)/((g1*p2*beta2+sigma2)*Gamma))

R4 = np.log10(1+(g2*p2)/((g2*p1*beta1+sigma2)*Gamma))

R3d_2 = R3+R4


fig = plt.figure()
ax = fig.gca(projection='3d')

# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple = ('y', 'b')
colors = np.empty(p1.shape, dtype=str)
for y in range(len(p2)):
    for x in range(len(p1)):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]
        
# Plot the surface.
surf = ax.plot_surface(p1, p2, R3d_1, cmap=cm.coolwarm, linewidth=0, antialiased=False, facecolor=colors)


#surf = ax.plot_surface(p1, p2, R3d_2, cmap=cm.coolwarm, linewidth=0, antialiased=False, facecolor='blue')

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

## rotate the axes and update
#for angle in range(0, 360):
    #ax.view_init(30, angle)
    #plt.draw()
    #plt.pause(.001)


# In[38]:




