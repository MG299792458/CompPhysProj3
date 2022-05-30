class crystal():
    """The growth of a crystal with dislocations.

    Parameter
    ---------
    dims : Tulple
        Dimensions of the crystal surface
    mu : float
        Dimensionless chemical potential
    T : float
        Dimensionless temperature
    """

    def __init__(self, dims, mu, T):
        self.T = T
        self.mu = mu
        self.dims = dims
        self.fx_matrix = np.zeros(dims)
        self.bx_matrix = np.zeros(dims)
        self.fy_matrix = np.zeros(dims)
        self.by_matrix = np.zeros(dims)
        self.num_dislocations = 0
        self.time = 0
        self.surface = np.ones(self.dims)
        self.surface = self.surface[:,:,np.newaxis]

    
    def dislocation_matrices(self, face, face_loc, boundaries, b):
        """Defining a  dislocation line on the (001) cystal surface.

        Parameter
        ---------
        face : int
        face_loc : int
        boundaries : Tulple
        b : int
        """

        line = np.arange(boundaries[0], boundaries[1], 1, dtype=int)
        dislocation_line = np.ones(boundaries[1]-boundaries[0])*b
        f_matrix = np.zeros(self.dims)
        b_matrix = np.zeros(self.dims)
        if face == 0:
            f_matrix[face_loc, line] = dislocation_line
            b_matrix[face_loc-1, line] = -dislocation_line
            self.fx_matrix += f_matrix
            self.bx_matrix += b_matrix
        elif face == 1:
            f_matrix[line, face_loc] = dislocation_line
            b_matrix[line, face_loc-1] = -dislocation_line
            self.fy_matrix += f_matrix
            self.by_matrix += b_matrix
        else:
            raise ValueWarning('Value for [face] should be either 0 or 1')
        self.num_dislocations += 1
        print('crystal surface with {} dislocations'.format(str(self.num_dislocations)))

    
    def num_dislocations(self):
        print('There are {} dislocations'.format(str(self.num_dislocations)))

    
    def scan_neighbours(self, surface, fx_neigh, fy_neigh, bx_neigh, by_neigh, loc):
        """Scanning how many neighbours an atom on the surface has.

        Parameter
        ---------
        surface : nd.array
            crystal surface
        fx_neigh : nd.array
            forward scanning matrix for the x direction
        fy_neigh : nd.array
            forward scanning matrix for the y direction
        bx_neigh : nd.array
            backward scanning matrix for the x direction
        by_neigh : nd.array
            backward scanning matrix for the y direction
        loc : Tulple
            location of the atom

        Return
        ------
        n : int
            number of neighbours of one atom at [loc]
        """

        if surface[loc] <= fx_neigh[(loc[0]+1) % dims[0], loc[1] % dims[1]]:
            n += 1
        if surface[loc] <= fy_neigh[loc[0] % dims[0], (loc[1]+1) % dims[1]]:
            n += 1
        if surface[loc] <= bx_neigh[(loc[0]-1) % dims[0], loc[1] % dims[1]]:
            n += 1
        if surface[loc] <= by_neigh[loc[0] % dims[0], (loc[1]-1) % dims[1]]:
            n += 1
        return n


    def neighbours(self):
        """Identifying the number of neighbours of each surface atom using periodic boundary
        conditions for a surface with a single dislocation.
        
        Return
        ------
        neighbours : nd.array
            the number of neighbours of each atom
        """

        dims = self.dims
        neighbours = np.zeros(dims)
        surface = self.surface[:,:,self.time]
        fx_neigh = surface + self.fx_matrix
        bx_neigh = surface + self.bx_matrix
        fy_neigh = surface + self.fy_matrix
        by_neigh = surface + self.by_matrix
        for i in range(dims[0]):
            for j in range(dims[1]):
                loc = (i,j)
                n = self.scan_neighbours(surface, fx_neigh, fy_neigh, bx_neigh, by_neigh, loc)
                neighbours[i,j] = n
        return neighbours


    def evaporation_rate(self, n):
        """The evaporation rate based on the number of neighbours and temperature.

        Parameter
        ---------
        n : int
            the number of neighbours of an atom

        Return
        ------
        k_minus : float
            the evaporation rate
        """

        k_minus = np.exp(-n*self.T)
        return k_minus


    def impingement_rate(self):
        """The impingement rate based on the chemical potential and temperature.

        Return
        ------
        k_plus : float
            the impingement rate
        """

        k_3 = self.evaporation_rate(3)
        k_plus = np.exp(self.mu)*k_3
        return k_plus


    def nn_migration_rate(self, n):
        """The surface migration rate from n to n.

        Parameter
        ---------
        n : int
            the number of neighbours of an atom
        """

        if n == 1:
            Esd = 1/2
        elif n == 2:
            Esd = 3/2
        else:
            Esd = 5/2
        k_nn = 1/8*np.exp(-Esd*self.T)
        return k_nn


    def nm_migration_rate(self, loc_n, loc_m, neighbours):
        """The surface migration rate of a single atom

        Parameter
        ---------
        loc_n : Tulple
            the original position of the atom
        loc_m : Tulple
            the new position of the atom
        neighbours : nd.array
            the number neighbours of the atoms for the original atom position

        Return
        ------
        k_nm : float
            the migration rate
        """

        n = neighbours[loc_n]
        new_surface = self.surface[:,:,self.time]
        new_surface[loc_n] += -1
        new_surface[loc_m] += 1
        fx_neighbour = new_surface + self.fx_matrix
        fy_neighbour = new_surface + self.fy_matrix
        bx_neighbour = new_surface + self.bx_matrix
        by_neighbour = new_surface + self.by_matrix
        m = self.scan_neighbours(new_surface, fx_neighbour, fy_neighbour, bx_neighbour, by_neighbour, loc_m)

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
    
    
    def choose_subset(self):
        """choose the number of neighbours each atom in the subset will have in which 
        interaction will occur."""

        T = self.T
        mu = self.mu
        counts = dict(zip([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]))
        neigh = self.neighbours()
        unique, counting = np.unique(neigh, return_counts = True)
        index = 0
        impingement_rate = self.impingement_rate()

        for number in unique:
            counts[number] = counting[index]
            index += 1

        denom = 0
        for i in range(1,6):
            denom += counts[i] * (self.evaporation_rate(i) + impingement_rate
                                  + self.nn_migration_rate(i))

        prob = np.zeros(5)
        for i in range(5):
            prob[i] = counts[i+1] * (self.evaporation_rate(i+1) + impingement_rate
                                     + self.nn_migration_rate(i+1)) / denom

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

        return subset, neigh


    def interaction(self):
        """Randomly lets interaction take place in chosen subset."""

        dims = self.dims
        surface = self.surface[:,:,self.time]
        change_surface = np.zeros(dims)
        subset, neigh = self.choose_subset()
        options_x = np.where(neigh==subset)[0]
        options_y = np.where(neigh==subset)[1]
        site = choice(range(np.size(options_x)))

        location = (options_x[site], options_y[site])

        k_plus = self.impingement_rate()
        k_minus = self.evaporation_rate(subset)
        k_nn = self.nn_migration_rate(subset)

        denom = k_plus + k_minus + k_nn
        #Speed of the code might be improved if new neighbour matrix is also constructed in this part
        rand = uniform(0,1)
        if rand < k_plus/denom:
            change_surface[location] += 1
            print('impingement')
        elif rand < (k_plus+k_minus)/denom:
            change_surface[location] -= 1
            print('evaporation')
        else:
            #Should the choise for migration location depend on k_nm????????????????????????????
            options = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
            migrate = choice(options)
            prob = self.nm_migration_rate(location, migrate, neigh)
            rand = uniform(0,1)
            if rand < prob:
                print('migration')
                change_surface[location] -= 1
                change_surface[(location[0]+migrate[0]) % dims[0], (location[1]+migrate[1]) % dims[1]] += 1

        new_surface = surface + change_surface
        #self.surface = np.append(self.surface, new_surface[:,:,np.newaxis], axis=2)
        #self.time += 1
        return new_surface
    
    
    
    
    
    
    
    
    
    
class grow_crystal():
    """The growth of a crystal with dislocations.
    
    Parameter
    ---------
    
    """
    def __init__(self, dims, mu, T):
        self.T = T
        self.mu = mu
        self.dims = dims
        self.fx_matrix = np.zeros(dims)
        self.bx_matrix = np.zeros(dims)
        self.fy_matrix = np.zeros(dims)
        self.by_matrix = np.zeros(dims)
        self.num_dislocations = 0
        self.time = 0
        self.surface = np.ones(self.dims)
        self.surface = self.surface[:,:,np.newaxis]
        self.neigh = np.array([])
    
    def dislocation_matrices(self, face, face_loc, boundaries, b):
        """Defining a  dislocation line on the (001) cystal surface."""
        
        line = np.arange(boundaries[0], boundaries[1], 1, dtype=int)
        dislocation_line = np.ones(boundaries[1]-boundaries[0])*b
        f_matrix = np.zeros(self.dims)
        b_matrix = np.zeros(self.dims)
        if face == 0:
            f_matrix[face_loc, line] = dislocation_line
            b_matrix[face_loc-1, line] = -dislocation_line
            self.fx_matrix += f_matrix
            self.bx_matrix += b_matrix
        elif face == 1:
            f_matrix[line, face_loc] = dislocation_line
            b_matrix[line, face_loc-1] = -dislocation_line
            self.fy_matrix += f_matrix
            self.by_matrix += b_matrix
        else:
            raise ValueWarning('Value for [face] should be either 0 or 1')
        self.num_dislocations += 1
        print('crystal surface with {} dislocations'.format(str(self.num_dislocations)))
    
    def num_dislocations(self):
        print('There are {} dislocations'.format(str(self.num_dislocations)))
    
    def scan_neighbours(self, surface, fx_neigh, fy_neigh, bx_neigh, by_neigh, loc):
        """Scanning how many neighbours an atom on the surface has."""
        
        dims = self.dims
        n = 1
        if surface[loc] <= fx_neigh[(loc[0]+1) % dims[0], loc[1] % dims[1]]:
            n += 1
        if surface[loc] <= fy_neigh[loc[0] % dims[0], (loc[1]+1) % dims[1]]:
            n += 1
        if surface[loc] <= bx_neigh[(loc[0]-1) % dims[0], loc[1] % dims[1]]:
            n += 1
        if surface[loc] <= by_neigh[loc[0] % dims[0], (loc[1]-1) % dims[1]]:
            n += 1
        return n
    
    def neighbours(self):
        """Identifying the number of neighbours of each surface atom using periodic boundary
        conditions for a surface with a single dislocation."""
        
        dims = self.dims
        neigh = np.zeros(dims)
        surface = self.surface[:,:,self.time]
        fx_neigh = surface + self.fx_matrix
        bx_neigh = surface + self.bx_matrix
        fy_neigh = surface + self.fy_matrix
        by_neigh = surface + self.by_matrix
        for i in range(dims[0]):
            for j in range(dims[1]):
                loc = (i,j)
                n = self.scan_neighbours(surface, fx_neigh, fy_neigh, bx_neigh, by_neigh, loc)
                neigh[i,j] = n
        return neigh
    
    def evaporation_rate(self, n):
        """The evaporation rate based on the number of neighbours and temperature."""
        
        k_minus = np.exp(-n*self.T)
        return k_minus
    
    def impingement_rate(self):
        """The impingement rate based on the chemical potential and temperature."""
        
        k_3 = self.evaporation_rate(3)
        k_plus = np.exp(self.mu)*k_3
        return k_plus
    
    def nn_migration_rate(self, n):
        if n == 1:
            Esd = 1/2
        elif n == 2:
            Esd = 3/2
        else:
            Esd = 5/2
        k_nn = 1/8*np.exp(-Esd*self.T)
        return k_nn
    
    def nm_migration_rate(self, loc_n, loc_m, neigh):
        n = neigh[loc_n]
        new_surface = self.surface[:,:,self.time]
        new_surface[loc_n] += -1
        new_surface[loc_m] += 1
        fx_neighbour = new_surface + self.fx_matrix
        fy_neighbour = new_surface + self.fy_matrix
        bx_neighbour = new_surface + self.bx_matrix
        by_neighbour = new_surface + self.by_matrix
        m = self.scan_neighbours(new_surface, fx_neighbour, fy_neighbour, bx_neighbour, by_neighbour, loc_m)
        
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
    
    def choose_subset(self):
        T = self.T
        mu = self.mu
        counts = dict(zip([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]))
        if self.time == 0:
            first_neigh = self.neighbours()
            self.neigh = first_neigh[:,:,np.newaxis]
        neigh = self.neigh[:,:,self.time]
        unique, counting = np.unique(neigh, return_counts = True)
        index = 0
        impingement_rate = self.impingement_rate()
        
        for number in unique:
            counts[number] = counting[index]
            index += 1

        denom = 0
        for i in range(1,6):
            denom += counts[i] * (self.evaporation_rate(i) + impingement_rate)
                                  #+ self.nn_migration_rate(i))

        prob = np.zeros(5)
        for i in range(5):
            prob[i] = counts[i+1] * (self.evaporation_rate(i+1) + impingement_rate)/denom
                                     #+ self.nn_migration_rate(i+1)) / denom

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
        
        return subset#, neigh
    
    def interaction(self):
        dims = self.dims
        surface = self.surface[:,:,self.time]
        change_surface = np.zeros(dims)
        subset = self.choose_subset()
        neigh = self.neigh[:,:,self.time]
        options_x = np.where(neigh==subset)[0]
        options_y = np.where(neigh==subset)[1]
        site = choice(range(np.size(options_x)))

        location = (options_x[site], options_y[site])

        k_plus = self.impingement_rate()
        k_minus = self.evaporation_rate(subset)
        k_nn = self.nn_migration_rate(subset)

        denom = k_plus + k_minus# + k_nn
        scan_loc_matrix = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
        change_neigh = np.zeros(dims)

        rand = uniform(0,1)
        if rand < k_plus/denom:
            change_surface[location] += 1
            new_surface = surface + change_surface
            for i in scan_loc_matrix: #create new neighbour matrix
                (location[0]+i[0], location[1]+i[1])
                scan_loc = ((location[0]+i[0]) % dims[0], (location[1]+i[1]) % dims[1])
                fx_neigh = new_surface + self.fx_matrix
                fy_neigh = new_surface + self.fy_matrix
                bx_neigh = new_surface + self.bx_matrix
                by_neigh = new_surface + self.by_matrix
                change_neigh[scan_loc] = self.scan_neighbours(new_surface, fx_neigh, 
                                                              fy_neigh, bx_neigh, 
                                                              by_neigh, scan_loc) - neigh[scan_loc]
            new_neigh = neigh + change_neigh
        elif rand < (k_plus+k_minus)/denom:
            change_surface[location] -= 1
            new_surface = surface + change_surface
            for i in scan_loc_matrix: #create new neighbour matrix
                scan_loc = ((location[0]+i[0]) % dims[0], (location[1]+i[1]) % dims[1])
                fx_neigh = new_surface + self.fx_matrix
                fy_neigh = new_surface + self.fy_matrix
                bx_neigh = new_surface + self.bx_matrix
                by_neigh = new_surface + self.by_matrix
                change_neigh[scan_loc] = self.scan_neighbours(new_surface, fx_neigh, 
                                                              fy_neigh, bx_neigh, 
                                                              by_neigh, scan_loc) - neigh[scan_loc]
            new_neigh = neigh + change_neigh
        else:
            options = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
            migrate = choice(options)
            prob = self.nm_migration_rate(location, migrate, neigh)
            rand = uniform(0,1)
            if rand < prob:
                change_surface[location] -= 1
                change_surface[(location[0]+migrate[0]) % dims[0], (location[1]+migrate[1]) % dims[1]] +=1
                new_surface = surface + change_surface
                for i in scan_loc_matrix:
                    scan_loc1 = ((location[0]+i[0]) % dims[0], (location[1]+i[1]) % dims[1])
                    scan_loc2 = ((migrate[0]+i[0]) % dims[0], (migrate[1]+i[1]) % dims[1])
                    fx_neigh = new_surface + self.fx_matrix
                    fy_neigh = new_surface + self.fy_matrix
                    bx_neigh = new_surface + self.bx_matrix
                    by_neigh = new_surface + self.by_matrix
                    change_neigh[scan_loc1] = self.scan_neighbours(new_surface, fx_neigh,
                                                                   fy_neigh, bx_neigh,
                                                                   by_neigh, scan_loc1) - neigh[scan_loc1]
                    change_neigh[scan_loc2] = self.scan_neighbours(new_surface, fx_neigh,
                                                                   fy_neigh, bx_neigh,
                                                                   by_neigh, scan_loc2) - neigh[scan_loc2]
                new_neigh = neigh + change_neigh
            else:
                new_neigh = neigh
        self.surface = np.append(self.surface, new_surface[:,:,np.newaxis], axis=2)
        self.neigh = np.append(self.neigh, new_neigh[:,:,np.newaxis], axis=2)
        self.time += 1
