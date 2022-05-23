# %%
import core_func as cf
import numpy as np
import matplotlib.pyplot as plt

# %%
surface = cf.init_crystal([50,50])

# %%
neigh = cf.nearest_neighbours(surface)

# %%
neigh

kb = 1.380649e-23
T = 3 * 5e12 / 4.9 / kb
print(T)


# %%
subset = cf.choose_subset(surface, T, 0)
'''phi and mu are not set correctly so T has to be chosen very high'''

# %%
for i in range(500):

    surface = cf.interaction(surface, T, 0)


# %%
surface

# %%
neigh = cf.nearest_neighbours(surface)

# %%
neigh

# %%
xr, yr = surface.shape
x, y = np.arange(xr+1), np.arange(yr+1)

plt.pcolor(x, y, surface)
plt.gca().set_aspect('equal')
plt.title("Surface deposition height")
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.colorbar()
plt.savefig("week1.png")
plt.show()

# %%



