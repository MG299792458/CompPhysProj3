import numpy as np
import matplotlib.pyplot as plt
from core_func import *
from data_func import *


growth_array = np.load("Simulation Data/Crystal_growth_mu=1.5_T=2_migration=False_N=1000000_b=0_dislocation=False_v1.npy")
mu = 1.5
T = 2

kplus = np.exp(mu)*evaporation_rate(3, T)
print('kplus {:.2f}'.format(kplus))

shape = growth_array.shape
iter = 1000000
print(shape)

size_x, size_y, steps = shape


iter_interval = iter / steps
rates = np.array([])
rates_err = np.array([])

for i in range(steps-1):
    start = i*iter_interval
    stop = (i+1)*iter_interval

    rate, err = find_rate(growth_array[:,:,i], growth_array[:,:,i+1], start, stop)
    rates = np.append(rates, rate)
    rates_err = np.append(rates_err, err)


intervals = np.arange(steps-1)*iter_interval
plt.scatter(intervals, rates/kplus)
plt.show()
