#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
from num_scheme import heat_eq as hq
import numpy as np


# Initializing numerical scheme
hq1 = hq()

x, t = hq1.get_x_t()

# Boundary- and initial conditions
f = g = 0 # BC 
phi = np.where(x <= 0.5, 2 * x, 2 * (1 - x))







if __name__ == '__main__':
    pass