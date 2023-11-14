#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
from num_scheme import heat_eq as hq
from heat_plot import Plot as pt
from analytic import Analytic


# Creating h(x, t) function
def H(expression, x:np.ndarray, t:np.ndarray):
    """
    Function h(x, t) used in analytiacal and numerical solutions
    """
    sol = np.zeros((x.shape[0], t.shape[0]))
    for n in range(sol.shape[1]):
        for i in range(sol.shape[0]):
            sol[i, n] = expression(x[i], t[n])
    return sol


# Initializing numerical scheme
hq1 = hq()

# Gettiing the position and time arrays
x, t = hq1.get_x_t()

# Computing h(x, t)
h = H(lambda x, t: x * (10 - x) * np.exp(-t), x, t)

# Boundary- and initial conditions
f = g = np.ones_like(t) # BC 
phi = np.where(x <= 0.5, 1 + 2 * x, 1 + 2 * (1 - x))
rho = np.zeros((x.shape[0], t.shape[0]))

# Running the numerical scheme
sol_1 = hq1.finite_diff(f, g, phi, rho)

# Plotting 
plt = pt(x, t, sol_1)

if __name__ == '__main__':
    # Plotting and animating solution of the heat equation
    # plt.plot(0, hq1.get_s())
    plt.animate(hq1.get_s())