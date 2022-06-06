import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_func import *
from core_func import *

dims = [20, 20]
T = 3.2
Migration = False
Disloc = True

face = 0
face_loc = 10
boundaries = [0, 10]
b = 2

N = 1000
dN = 10000

mu_min = 0
mu_max = 6
dmu = 0.5
mu = np.arange(mu_min, mu_max+dmu, dmu)

crystal = grow_crystal(dims, mu, T, Migration)
crystal.dislocation_matrices(face, face_loc, boundaries, b)
for i in range(N):
    crystal.interaction(dN)

plot_surface(crystal.surface)


exit()
