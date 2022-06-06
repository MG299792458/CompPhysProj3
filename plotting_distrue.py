import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_func import *
from core_func import *
from os import listdir
import os.path as osp
from scipy.optimize import curve_fit as cv


simpath = 'Simulation Data\\dislocation false\\T 4'
path = osp.join(osp.dirname(__file__), simpath)

files = [osp.join(path,file) for file in listdir(path)]

simulations = [Simulation(file) for file in files]
simulations_amnt = len(simulations)


for simulation in simulations:
    sim_rate, sim_rate_err = compute_rates(simulation=simulation)
    mu_over_T = simulation.parameters['mu']

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
err2 = []

for simulation in simulations:
    kplus = simulation.parameters['kplus']
    rate = np.average(simulation.rates[-20:])
    err = np.average(simulation.rates_err[-20:])
    mu_over_T = simulation.parameters['mu/T']

    muTs2.append(mu_over_T)
    rates2.append(rate)
    err2.append(err)


def growth_analytical(mu, gamma):
    rate = (np.pi/3)**(1/3) * mu**(1/6) * (1 - np.exp(-mu))**(2/3)\
         * np.exp(-4/3 * gamma**2 /mu)

    return rate

copt, ccov = cv(growth_analytical, muTs2, rates2)

mu_an = np.linspace(0,3.5,100)
rate_an = growth_analytical(mu_an, copt)

plt.scatter(muTs2, rates2, label=r"simul.")
plt.plot(mu_an, rate_an, label=r"fit. $\gamma = $"+"{:.2f}".format(copt[0]))
# plt.errorbar(muTs2, rates2, yerr=err2)
plt.legend(frameon=False)
plt.ylabel(r'$R/k^+$')
plt.xlabel(r'$\mu$')
plt.savefig('dislocations.png')
plt.show()
