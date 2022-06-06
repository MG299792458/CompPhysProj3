import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_func import *
from core_func import *
from os import listdir
import os.path as osp
from scipy.optimize import curve_fit as cv


simpath = 'Simulation Data\\dislocation false\\T 2'
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

simpath = 'Simulation Data\\dislocation true\\T 2'
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
muTsd = []
ratesd = []
errd = []

for simulation in simulations:
    kplus = simulation.parameters['kplus']
    rate = np.average(simulation.rates[-20:])
    err = np.average(simulation.rates_err[-20:])
    mu_over_T = simulation.parameters['mu/T']

    muTsd.append(mu_over_T)
    ratesd.append(rate)
    errd.append(err)

def growth_analytical_perfect(mu, gamma):
    rate = (np.pi/3)**(1/3) * mu**(1/6) * (1 - np.exp(-mu))**(2/3)\
         * np.exp(-4/3 * gamma**2 /mu)

    return rate

def growth_analytical_spiral(mu, gamma, phi):
    rate = (np.pi/3)**(1/3) * mu**(1/6) * (1 - np.exp(-mu))**(2/3)\
         * np.exp(-4/3 * gamma**2 /mu)

    spir_rate = (0.053*2*mu*phi*(1-np.exp(mu)))

    return rate+spir_rate


copt, ccov = cv(growth_analytical_perfect, muTs2, rates2)
copt0, ccov0 = cv(growth_analytical_perfect, muTsd, ratesd)

mu_an = np.linspace(0,3.5,100)
rate_an_per = growth_analytical_perfect(mu_an, copt)
rate_an_spi = growth_analytical_perfect(mu_an, *copt0)

plt.scatter(muTs2, rates2, label=r"per.")
plt.scatter(muTsd, ratesd, label=r'spiral' )
plt.plot(mu_an, rate_an_per, label=r"fit. $\gamma = $"+"{:.2f}".format(copt[0]))
plt.plot(mu_an, rate_an_spi, label=r"fit. $\gamma = $"+"{:.2f}".format(copt0[0]))
# plt.errorbar(muTs2, rates2, yerr=err2)
plt.legend(frameon=False)
plt.ylabel(r'$R/k^+$')
plt.xlabel(r'$\mu$')
plt.title('FIg 8 in paper')
plt.savefig('comparing_spir_perfect_t=2_atoms_fits.png')
plt.show()
