
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from matplotlib.widgets import Slider, Button

#Setup figure and data
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
delta = 0.5

t = np.arange(0.0, 100.0, 0.1)
x = np.arange(-3.0, 4.001, delta)
y = np.arange(-4.0, 3.001, delta)

X, Y = np.meshgrid(x, y)

Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = (Z1 - Z2) * 5.

cmap = plt.cm.rainbow
im = ax.pcolormesh(X, Y, Z, cmap=cmap)
fig.colorbar(im)
axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
stime = Slider(axtime, 'Time', 0.0, 100.0, valinit=50.0)
slider_beta = Slider(axtime, 'Beta', 0.0, 1.0, valinit=0.0)

#Routines to reset and update sliding bar
def reset(event):
    stime.reset()

def update(val):
    time = stime.val/10.
    Z = (Z1 - Z2) * time
    im.set_array(Z.ravel())
    fig.canvas.draw()

def update_R(val):
    beta = val
    


#Bind sliding bar and reset button
stime.on_changed(update)
slider_beta.on_changed(update_R)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
button.on_clicked(reset)

plt.show()

def throughput(beta):

    # Channel conditions
    g1 = 2.5
    g2 = 1.5
    p1_max = 10
    p2_max = 100
    sigma2 = 1
    Gamma = 1

    # Make data.
    p1 = np.linspace(0, p1_max, 100)
    p2 = np.linspace(0, p2_max, 100)
    p1, p2 = np.meshgrid(p1, p2)

    R1 = np.log2(1 + (g1 * p1) / ((g1 * p2 * beta + sigma2) * Gamma))
    R2 = np.log2(1 + (g2 * p2) / ((g2 * p1 * beta + sigma2) * Gamma))
    R3d = R1 + R2
    return R3d