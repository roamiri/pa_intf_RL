
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from matplotlib.widgets import Slider, Button

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib, time

class plot3dClass( object ):

    def __init__( self):
        # Channel conditions
        self.g1 = 2.5
        self.g2 = 1.5
        self.p1_max = 10
        self.p2_max = 100
        self.sigma2 = 1
        self.Gamma = 1

        # Make data.
        self.p1 = np.linspace(0, self.p1_max, 100)
        self.p2 = np.linspace(0, self.p2_max, 100)
        self.X, self.Y = np.meshgrid(self.p1, self.p2)

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.fig.subplots_adjust(bottom=0.25)

        # self.fig, self.ax = plt.subplots()
        # plt.subplots_adjust(bottom=0.25)

        self.ax.w_zaxis.set_major_locator( LinearLocator( 10 ) )
        self.ax.w_zaxis.set_major_formatter( FormatStrFormatter( '%.03f' ) )

        self.Z = self.throughput(0.0)

        self.surf = self.ax.plot_surface(self.p1, self.p2, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False )

        self.axcolor = 'lightgoldenrodyellow'
        self.ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=self.axcolor)
        self.slider_beta = Slider(self.ax_slider, 'Beta', 0.0, 1.0, valinit=0.0)
        self.slider_beta.on_changed(self.drawNow)
        self.fig.colorbar(self.surf, shrink=0.5, aspect=5)
        self.fig.show() # maybe you want to see this frame?

    def throughput(self,beta):
        R1 = np.log2(1 + (self.g1 * self.p1) / ((self.g1 * self.p2 * beta + self.sigma2) * self.Gamma))
        R2 = np.log2(1 + (self.g2 * self.p2) / ((self.g2 * self.p1 * beta + self.sigma2) * self.Gamma))
        R3d = R1 + R2
        return R3d

    def drawNow( self, val):
        self.Z = self.throughput(val)
        self.surf.remove()
        self.surf = self.ax.plot_surface(self.p1, self.p2, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False )
        plt.draw()                      # redraw the canvas
        # time.sleep(1)

# matplotlib.interactive(True)
# p = plot3dClass()
# while(1):
#     p.fig.show()

g1 = 2.5
g2 = 1.5
p1_max = 10
p2_max = 20
sigma2 = 1
Gamma = 1

# Make data.
p1 = np.linspace(0, p1_max, 100)
p2 = np.linspace(0, p2_max, 100)
X, Y = np.meshgrid(p1, p2)

fig = plt.figure()
plt.subplots_adjust(bottom=0.25)
ax = fig.gca(projection='3d')

def throughput(beta):
    R1 = np.log2(1 + (g1 * X) / ((g1 * Y * beta + sigma2) * Gamma))
    R2 = np.log2(1 + (g2 * Y) / ((g2 * X * beta + sigma2) * Gamma))
    R3d = R1 + R2
    return R3d

Z = throughput(0.0)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
slider_beta = Slider(ax_slider, 'Beta', 0.0, 1.0, valinit=0.0)

def update_R(val):
    Z = throughput(val)
    ax.clear()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

slider_beta.on_changed(update_R)

plt.show()

#Bind sliding bar and reset button
# stime.on_changed(update)

# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# button.on_clicked(reset)

# plt.show()

