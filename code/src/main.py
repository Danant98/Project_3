#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
from num_scheme import heat_eq as hq
from heat_plot import Plot as pt
from analytic import Analytic
from fourier import Fourier as f



def source_IC(x:np.ndarray, t:np.ndarray, h:np.ndarray):
    """
    Computing the source term and initial condition 
    for 1b)
    """
    # Initializing the source and initial condition arrays
    rho = np.zeros((x.shape[0], t.shape[0]))
    phi = np.zeros((x.shape[0], t.shape[0]))


# Initializing numerical scheme
hq1 = hq()

# Gettiing the position and time arrays
x, t = hq1.get_x_t()

# Computing h(x, t)
h = np.ones((x.shape, t.shape))

# Boundary- and initial conditions for first example
f_1 = np.sin(t)
g_1 = np.zeros_like(t) # BC  
phi_1 = x * (1 - x)
rho_1 = np.zeros((x.shape[0], t.shape[0]))

# Boundary conditions for second example
f_2 = g_2 = np.zeros_like(t)
rho_2 = np.exp(-t) * np.ones((x.shape[0], t.shape[0]))

# Running the numerical scheme
sol_1 = hq1.finite_diff(f_1, g_1, phi_1, rho_1)
sol_2 = hq1.finite_diff(f_2, g_2, phi_1, rho_2)

# Computing the fourier solution
f_1 = f.fourier_1(x, t)
f_2 = f.fourier_2(x, t)

# Running plotting class for both the numerical solutions
plt = pt(x, t, sol_1)
plt2 = pt(x, t, sol_2)

if __name__ == '__main__':
    pass
    # Plotting and animating solution of the heat equation
    # plt.plot(0, hq1.get_s())
    # plt.animate(hq1.get_s())
    # plt.animate(hq1.get_s())
    # plt.rmse_plot(f_1, sol_1, label = 'f1')
    # plt.rmse_plot(f_2, sol_2, label = 'f2')