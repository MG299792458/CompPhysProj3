# Simulating growth on crystal interfaces
The contents of this repository allow for the simulation of growth on both perfect and imperfect crystal faces.

## Table of contents
[[_TOC_]]

## Installation
The module can be installed using pip or can be imported when the working file is placed in the same directory.

```python
from data_func import *
from core_func import *
```
to access the necessary functions and classes.

## Usage

```python

#Define physical parameters
dims = [80, 80]
T = 3.2
mu = 1.5
Disloc = True
Migration = False
b = 2

# Define simulation parameters
N = 100000
dN = 10000

# Initialise a crystal
crystal = grow_crystal(dims, mu_i, T, Migration)

# Introduce a defect, this is optional
face = 0
face_loc = 40
boundaries = [0, 40]
crystal.dislocation_matrices(face, face_loc, boundaries, b)

# Run as many interactions as you want
for i in range(N):
    crystal.interaction(dN)

```

## Contributors

@sangersjeroen
@agefrancke2
@mwglorie
