from matplotlib import projections
from matplotlib.animation import Animation, FuncAnimation
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_func import *
from core_func import *
import os.path as osp
from os import listdir

mpl.rcParams['figure.dpi'] = 120

lght = mpl.colors.LightSource(azdeg=-30, altdeg=30)

dims = [50, 50]
T = 6.5
Migration = False
Disloc = True

face = 0
face_loc = 25
boundaries = [0, 15]
b = 2

N = 100000
dN = 1000

mu = 6

crystal = grow_crystal(dims, mu, T, Migration)
crystal.dislocation_matrices(face, face_loc, boundaries, b)
for i in range(N):
    crystal.interaction(dN)

surface = crystal.surface

plt.imshow(surface); plt.colorbar()
plt.show()

test = True

def get_volume(surface):
    data = surface.copy()
    height_min = np.min(data)

    data = data - (height_min-1)
    height_min = (np.min(data)).astype('int')
    height_max = (np.max(data)).astype('int')
    height = (height_max - height_min).astype('int')
    shape = data.shape

    volume = np.ones((shape[0],shape[1],height)) - 1

    for i in range(shape[0]):
        for j in range(shape[1]):
            fill = data[i,j]
            volume[i,j,0:int(fill)] += 1

    colors = np.empty(volume.shape + (3,), dtype='object')
    colors[..., 0] = 40/256
    colors[..., 1] = 255/256
    colors[..., 2] = 255/256

    x, y, h = np.indices((shape[0]+1, shape[1]+1, height+1))
    h = h/20

    return [x, y, h], volume, colors

fig = plt.figure(figsize=(4,3))
ax = fig.gca(projection='3d')
ax.set_axis_off()
ax.view_init(elev=30, azim=-30)

surf0 = crystal.N_surface[:,:,1]
coord0, vol0, col0 = get_volume(surf0)


def init():
    ax.voxels(*coord0, vol0, lightsource=lght, edgecolors='k', linewidth=0.5)
    return fig,

def animate(i):
    print(i)
    ax.clear()
    surf = crystal.N_surface[:,:,i+1]
    coord, vol, color = get_volume(surf)
    ax.voxels(*coord, vol, lightsource=lght, facecolor='white', edgecolor='k', linewidth=0.5)
    ax.view_init(elev=30, azim=-30)
    return fig,

animation = FuncAnimation(fig, animate, init_func=init, frames=len(crystal.N_surface[0,0,:])-1, interval=25)

animation.save('animation.mp4', fps=10)

exit()
