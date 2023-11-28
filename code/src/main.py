#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
from num_scheme import heat_eq as hq
from heat_plot import Plot as pt
from analytic import Analytic
from fourier import Fourier as f


# Computing rho(x, t)
def source(x:np.ndarray, h:np.ndarray, h_t:np.ndarray, h_x:np.ndarray, h_xx:np.ndarray,
           g:np.ndarray, f:np.ndarray, g_t:np.ndarray, f_t:np.ndarray, l:float = 1):
    """
    Computing the source term and initial condition 
    for 1b)
    """
    # Initializing the source array
    rho = np.zeros_like(h)
    for n in range(h.shape[1]):
        for i in range(x.shape[0]):
            # Computing v(x, t)
            v = x[i] * (g[n] / h[-1, n]) + (l - x[i]) * (f[n] / h[0, n])
            # Computing v_t(x, t)
            v_t_term_1 = x[i] * (((g_t[n] * h[-1, n]) - (h_t[-1, n] * g[n])) / h[-1, n]**2)
            v_t_term_2 = (l - x[i]) * (((f_t[n] * h[0, n]) - (h_t[0, n] * f[n])) / h[0, n]**2)
            v_t = v_t_term_1 + v_t_term_2
            A = (g[n] / h[-1, n]) - (f[n] / h[0, n])
            # Computing pho(x, t)
            term_1 = (1 / l) * (h_t[i, n] * v + h[i, n] * v_t)
            term_2 = (1 / l) * (h_xx[i, n] * v + 2 * h_x[i, n] * A)
            rho[i, n] = term_1 + term_2
    return rho


# Initializing numerical scheme
hq1 = hq()

# Gettiing the position and time arrays
x, t = hq1.get_x_t()

# Computing h(x, t)
h = np.exp(-t) * np.ones((x.shape[0], t.shape[0]))

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

# Running the analytical solution 
analytic_1 = Analytic(x, t, 1).solve(h, g_1, f_1)

# Computing the fourier solution
f_1 = f.fourier_1(x, t)
f_2 = f.fourier_2(x, t)

# Running plotting class for both the numerical solutions
plt = pt(x, t, sol_1)


if __name__ == '__main__':
    pass
    # Plotting and animating solution of the heat equation
    # plt.plot(0, hq1.get_s())
    plt.animate(sol_2, hq1.get_s())
    # plt.animate(hq1.get_s())
    # plt.animate(f_2)
    # plt.rmse_plot(f_1, sol_1, label = 'f1')
    # plt.rmse_plot(f_2, sol_2, label = 'f2')