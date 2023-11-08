#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
from num_scheme import heat_eq as hq
import numpy as np
from heat_plot import Plot as p

# Initializing numerical scheme
hq1 = hq()

# Gettiing the position and time arrays
x, t = hq1.get_x_t()

# Boundary- and initial conditions
f = g = 0 # BC 
phi = np.where(x <= 0.5, 2 * x, 2 * (1 - x))
rho = np.zeros((x.shape[0], t.shape[0]))

# Running the numerical scheme
sol_1 = hq1.finite_diff(f, g, phi, rho)

# Plotting 
plt = p(x, t, sol_1)

if __name__ == '__main__':
    # plt.plot(0, hq1.get_s())
    plt.animate(hq1.get_s())