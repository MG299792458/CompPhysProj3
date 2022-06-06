import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_func import *
from core_func import *
from os import listdir
import os.path as osp


simpath = 'Simulation Data\\dislocation true'
path = osp.join(osp.dirname(__file__), simpath)

files = [osp.join(path,file) for file in listdir(path)]

simulations = [Simulation(file) for file in files]
simulations_amnt = len(simulations)


for simulation in simulations:
    sim_rate, sim_rate_err = compute_rates(simulation=simulation)
    mu_over_T = simulation.parameters['mu']/simulation.parameters['T']

    temperature = simulation.parameters['T']
    mu = simulation.parameters['mu']

    kplus = np.exp(mu)*evaporation_rate(3, temperature)
    simulation.parameters['kplus'] = kplus

    simulation.rates = sim_rate
    simulation.rates_err = sim_rate_err
    simulation.parameters['mu/T'] = mu_over_T
    print(simulation.parameters['L']/simulation.parameters['T'])


kplusses = []
muTs2 = []
rates2 = []

for simulation in simulations:
    kplus = simulation.parameters['kplus']
    rate = np.average(simulation.rates[:])
    err = np.average(simulation.rates_err[:])
    mu_over_T = simulation.parameters['mu/T']

    muTs2.append(mu_over_T)
    rates2.append(rate)


plt.scatter(muTs2, rates2, label=r"T=4")
plt.legend(frameon=False)
plt.ylabel(r'$R/k^+$')
plt.xlabel(r'$\mu /k_b T$')
plt.savefig('dislocations.png')
plt.show()
