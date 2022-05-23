import numpy as np
import matplotlib.pyplot as plt
from random import uniform, choice
'''global parameters'''
kb = 1.380649e-23


def init_crystal(dims):
    """Create the initial crystal surface
    Creates a N x M surface with all lattice points occupied
    
    Parameter
    ---------
    dims : Tulple or nd.array
        The dimensions of the initial cristal surface
    
    Return
    ------
    surface : nd.array
        The occupied crystal latticle points
    """
    surface = np.ones(dims)
    return surface


def nearest_neighbours(surface):
    dims = surface.shape
    neighbours = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            if surface[i,j] <= surface[int(i+1-dims[0]*np.floor((i+1)/dims[0])),j]:
                neighbours[i,j] += 1
            if surface[i,j] <= surface[i,int(j+1-dims[1]*np.floor((j+1)/dims[1]))]:
                neighbours[i,j] += 1
            if surface[i,j] <= surface[int(i-1-dims[0]*np.floor((i-1)/dims[0])),j]:
                neighbours[i,j] += 1
            if surface[i,j] <= surface[i,int(j-1-dims[1]*np.floor((j-1)/dims[1]))]:
                neighbours[i,j] += 1
    return neighbours


def evaporation_rate(n, T):
    '''
    parameters:
        n : number of nearest neigbours
        T: temperature

    '''

    phi = 4.9/3
    '''not sure what value this is supposed to have'''
    mu = 5e12
    '''frequency factor again not sure what it is supposed to be'''
    k_minus = mu*np.exp(-n*phi/(kb*T))
    return k_minus


def impingement_rate(mu, T):

    k_plus = np.exp(mu/(kb*T))*evaporation_rate(3, T)
    return k_plus


def surface_migration_rate(n, m, T):

    phi = 4.9/3
    mu = 5e12

    if n == 1 or m == 1:
        Esd = phi/2
    elif n == 2 or m == 2:
        Esd = 3*phi/2
    else:
        Esd = 5*phi/2

    if m <= n:
        DeltaE = (n-m)*phi
    else:
        DeltaE = 0

    k_nm = 1/8*mu*np.exp(-(Esd+DeltaE)/(kb*T))
    return k_nm


def choose_subset(surface, T, mu):
    '''choose the number of neighbours each atom in the subset will have in which interaction will occur'''


    counts = dict(zip([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]))
    neigh = nearest_neighbours(surface)
    unique, counting = np.unique(neigh, return_counts = True)
    index = 0
    for number in unique:
        counts[number] = counting[index]
        index += 1

    denom = 0
    for i in range(1,6):
        denom += counts[i]*(evaporation_rate(i,T)+impingement_rate(mu, T)+surface_migration_rate(i,i,T))

    prob = np.zeros(5)
    for i in range(5):
        prob[i] = counts[i+1]*(evaporation_rate(i+1,T)+impingement_rate(mu, T)+surface_migration_rate(i+1,i+1,T))/denom

    rand = uniform(0,1)
    if rand < prob[0]:
        subset = 1
    elif rand < prob[0] + prob[1]:
        subset = 2
    elif rand < prob[0] + prob[1] + prob[2]:
        subset = 3
    elif rand < prob[0] + prob[1] + prob[2] + prob[3]:
        subset = 4
    elif rand < prob[0] + prob[1] + prob[2] + +prob[3] + prob[4]:
        subset = 5

    return subset


def interaction(surface, T, mu):
    '''randomly lets interaction take place in chosen subset'''

    dims = surface.shape
    neigh = nearest_neighbours(surface)
    subset = choose_subset(surface, T, mu)
    options_x = np.where(neigh==subset)[0]
    options_y = np.where(neigh==subset)[1]
    site = choice(range(np.size(options_x)))

    location = (options_x[site], options_y[site])

    k_plus = impingement_rate(mu, T)
    k_minus = evaporation_rate(subset, T)
    k_nn = surface_migration_rate(subset, subset, T)

    denom = k_plus + k_minus + k_nn

    rand = uniform(0,1)
    if rand < k_plus/denom:
        surface[location] += 1
    elif rand < (k_plus+k_minus)/denom:
        surface[location] -= 1
    else:
        migrate = choice([(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)])
        m = neigh[int(location[0]+migrate[0]-dims[0]*np.floor((location[0]+migrate[0])/dims[0])),
                  int(location[1]+migrate[1]-dims[1]*np.floor((location[1]+migrate[1])/dims[1]))]
        n = neigh[location]
        prob = surface_migration_rate(n, m, T)
        rand = uniform(0,1)
        if rand < prob:
            surface[location] -= 1
            surface[int(location[0]+migrate[0]-dims[0]*np.floor((location[0]+migrate[0])/dims[0])),
                  int(location[1]+migrate[1]-dims[1]*np.floor((location[1]+migrate[1])/dims[1]))] += 1


    return surface





