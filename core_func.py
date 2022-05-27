import numpy as np
import matplotlib.pyplot as plt
from random import uniform, choice
"""global parameters"""
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
    """Identifying the number of neighbours of each surface atom using periodic boundary
    conditions.

    Parameter
    ---------
    surface : nd.array
        An N x N matrix representing the surface of a crystal

    Return
    ------
    neighbours : nd.array
        An N x N matrix representing the number of neighbouring spaces of location (i, j)
        of the crystal surface that are occupied by an atom
    """
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
    """The evaporation rate based on the number of neighbours and temperature

    Parameter
    ---------
    n : int
        Number of nearest neigbours
    T : float
        Dimensionless temperature

    Return
    ------
    k_minus : float/nd.array
        Dimensionless evaporation rate
    """
    k_minus = np.exp(-n*T)
    return k_minus


def impingement_rate(mu, T):
    """The impingement rate based on the chemical potential and temperature

    Parameter
    ---------
    mu : float
        Dimensionless chemical potential
    T : float
        Dimensionless temperature

    Return
    ------
    k_plus : float
        Dimensionless impingement rate
    """

    k_3 = evaporation_rate(3, T)
    k_plus = np.exp(mu)*k_3
    return k_plus


def surface_migration_rate(n, m, T):
    """
    Parameter
    ---------
    n : int
        Number of neighbours of the selected atom
    m : int
        Number of neighbours of the neighbour of the selected particle
    T : float
        Dimensionless temperature

    Return
    ------
    k_nm : float
        Dimensionless migration rate
    """

    if n == 1 or m == 1:
        Esd = 1/2
    elif n == 2 or m == 2:
        Esd = 3/2
    else:
        Esd = 5/2

    if m <= n:
        DeltaE = n-m
    else:
        DeltaE = 0

    k_nm = 1/8*np.exp(-(Esd+DeltaE)*T)
    return k_nm


def choose_subset(surface, T, mu):
    """choose the number of neighbours each atom in the subset will have in which interaction will occur"""


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
    """randomly lets interaction take place in chosen subset"""

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



def dislocation_matrices(dims, face, face_loc, boundaries, b):
    """Defining a single dislocation line on the (001) cystal surface.
    
    Parameter
    ---------
    dims : int
        Dimensions of the crystal surface, with the surface dimensions dims x dims 
    face : int --> {0, 1}
        The plain the dislocation line is located in
        Value
        -----
        0 : dislocation line lies in the (100) plain
        1 : dislocation line lies in the (010) plain
    face_loc : int --> {1:dims-1}
        The location of the plane the dislocation line lies in. For
        [face_loc] = n, the dislocation is between the (n-1)th and nth
        atom.
    boundaries : tulple --> [start, end]
        The boundaries of the dislocation line with [start] < [end]
        Value
        -----
        start : {0:dims-1}
        end : {1:dims}
    b : int
        The magnitude of the Burgers vector
        If b=0, there is no dislocation
        If b>0, the step will go up
        If b<0, the step will go down
    
    Return
    ------
    forward_matrix : nd.array
        Matrix used to create dislocation when looking at the forward neighbour
    backward_matrix : nd.array
        Matrix used to create dislocation when looking at the backward neighbour
    """
    forward_matrix = np.zeros([dims,dims])
    backward_matrix = np.zeros([dims,dims])
    line = np.arange(boundaries[0], boundaries[1], 1, dtype=int)
    dislocation_line = np.ones(boundaries[1]-boundaries[0])*b
    if face == 0:
        forward_matrix[face_loc, line] = dislocation_line
        backward_matrix[face_loc-1, line] = -dislocation_line
    elif face == 1:
        forward_matrix[line, face_loc] = dislocation_line
        backward_matrix[line, face_loc-1] = -dislocation_line
    else:
        raise ValueWarning('Value for [face] should be either 0 or 1')
    
    return forward_matrix, backward_matrix


def dislocation_neighbours(surface, face, forward_matrix, backward_matrix):
    """Identifying the number of neighbours of each surface atom using periodic boundary
    conditions for a surface with a single dislocation.
    
    Parameter
    ---------
    surface : nd.array
        An N x N matrix representing the surface of a crystal
    face : int --> {0, 1}
        The plain the dislocation line is located in
        Value
        -----
        0 : dislocation line lies in the (100) plain
        1 : dislocation line lies in the (010) plain
    forward_matrix : nd.array
        Matrix used to create dislocation when looking at the forward neighbour
    backward_matrix : nd.array
        Matrix used to create dislocation when looking at the backward neighbour
    
    Return
    ------
    neighbours : nd.array
        An N x N matrix representing the number of neighbouring spaces of location (i, j)
        of the crystal surface that are occupied by an atom
    """
    neighbours = np.zeros(dims)
    forward_neighbour = surface + forward_matrix
    backward_neighbour = surface + backward_matrix
    
    if face == 0:
        for i in ranged(dims[0]):
            for j in range(dims[1]):
                if surface[i,j] <= forward_neighbour[int(i+1-dims[0]*np.floor((i+1)/dims[0])),j]:
                    neighbours[i,j] += 1
                if surface[i,j] <= surface[i,int(j+1-dims[1]*np.floor((j+1)/dims[1]))]:
                    neighbours[i,j] += 1
                if surface[i,j] <= backward_neighbour[int(i-1-dims[0]*np.floor((i-1)/dims[0])),j]:
                    neighbours[i,j] += 1
                if surface[i,j] <= surface[i,int(j-1-dims[1]*np.floor((j-1)/dims[1]))]:
                    neighbours[i,j] += 1
    elif face == 1:
        for i in ranged(dims[0]):
            for j in range(dims[1]):
                if surface[i,j] <= surface[int(i+1-dims[0]*np.floor((i+1)/dims[0])),j]:
                    neighbours[i,j] += 1
                if surface[i,j] <= forward_neighbour[i,int(j+1-dims[1]*np.floor((j+1)/dims[1]))]:
                    neighbours[i,j] += 1
                if surface[i,j] <= surface[int(i-1-dims[0]*np.floor((i-1)/dims[0])),j]:
                    neighbours[i,j] += 1
                if surface[i,j] <= backward_neighbour[i,int(j-1-dims[1]*np.floor((j-1)/dims[1]))]:
                    neighbours[i,j] += 1
    else:
        raise ValueWarning('Value for the face of the dislocation should be either 0 for (010) plane or 1 for the (100) plane')

