#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
from num_scheme import heat_eq as hq
from heat_plot import Plot as pt
from analytic import Analytic
from fourier import Fourier as f
from sklearn.metrics import mean_squared_error


# Computing rho(x, t)
def source(x:np.ndarray, h:np.ndarray, h_t:np.ndarray, h_x:np.ndarray, h_xx:np.ndarray,
           g:np.ndarray, f:np.ndarray, g_t:np.ndarray, f_t:np.ndarray, l:float = 1):
    """
    Computing the source term and initial condition 
    for 1b)
    """
    # Initializing the source array
    rho = np.zeros(h.shape)
    for n in range(rho.shape[1]):
        for i in range(rho.shape[0]):
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


# Computing h(x, t) = 1
# h = np.ones((x.shape[0], t.shape[0]))
# h_t = np.zeros((x.shape[0], t.shape[0]))
# h_x = np.zeros((x.shape[0], t.shape[0]))
# h_xx = np.zeros((x.shape[0], t.shape[0]))

# Computing h(x, t) = exp(t)
# h2 = np.exp(-t) * np.ones((x.shape[0], t.shape[0]))
# h2_t = -np.exp(-t) * np.zeros((x.shape[0], t.shape[0]))
# h2_x = np.zeros((x.shape[0], t.shape[0]))
# h2_xx = np.zeros((x.shape[0], t.shape[0]))

# Boundary- and initial conditions for first example
# f_1 = np.sin(t)
# g_1 = np.zeros_like(t) # BC
# # Derivative of f(t)
# f_1_t = np.cos(t)

# Exponenetial decaying boundaries for h(x, t)
# f_2 = g_2 = np.exp(-t)
# f_2_t = g_2_t = -np.exp(-t)

# Running the analytical solution 
# analytic_1 = Analytic(x, t).solve(h, g_1, f_1)
# analytic_2 = Analytic(x, t).solve(h2, g_2, f_2)

# phi_1 = analytic_1[:, 0]
# rho_1 = source(x, h, h_t, h_x, h_xx, g_1, f_1, g_1, f_1_t)

# rho_2 = source(x, h2, h2_t, h2_x, h2_xx, g_2, f_2, g_2_t, f_2_t)
# phi_2 = analytic_2[:, 0]

# Running the numerical scheme for h(x, t) = 1
# sol_1 = hq1.finite_diff(f_1, g_1, phi_1, rho_1)

# Running numerical scheme for h(x, t) = exp(-t)
# sol_2 = hq1.finite_diff(f_2, g_2, phi_2, rho_2)

# # Computing rmse for h(x, t) = 1
# rmse_1 = np.zeros_like(t)
# rmse_2 = np.zeros_like(t)
# for n in range(t.shape[0]):
#     # rmse_1[n] = mean_squared_error(sol_1[:, n], analytic_1[:, n])
#     rmse_2[n] = mean_squared_error(sol_2[:, n], analytic_2[:, n])


# ------------------------------------------------------------------------------------------------------
# Computing the fourier solution
# four_1, num = f.fourier_1(x, t, n = 7)
four_2, num = f.fourier_2(x, t, n = 9)

# finite diff soltution
# finite_f_1 = np.sin(t)
# finite_g_1 = np.zeros((t.shape[0]))
# finite_rho_1 = np.zeros((x.shape[0], t.shape[0]))
# finite_phi_1 = x * (1 - x)
# finite_sol_1 = hq1.finite_diff(finite_f_1, finite_g_1, finite_phi_1, finite_rho_1)

# Finite diff solution 2
finite_f_2 = finite_g_2 = np.zeros((t.shape[0]))
finite_rho_2 = np.exp(-t) * np.ones((x.shape[0], t.shape[0]))
finite_phi_2 = x * (1 - x)
finite_sol_2 = hq1.finite_diff(finite_f_2, finite_g_2, finite_phi_2, finite_rho_2)


# rmse_fourier_1 = np.zeros((t.shape[0]))
rmse_fourier_2 = np.zeros((t.shape[0]))
for n in range(t.shape[0]):
    # rmse_fourier_1[n] = mean_squared_error(finite_sol_1[:, n], four_1[:, n])
    rmse_fourier_2[n] = mean_squared_error(finite_sol_2[:, n], four_2[:, n])


# Running plotting class for both the numerical solutions
plt = pt(x, t)


if __name__ == '__main__':
    pass
    # Plotting and animating solution of the heat equation
    # plt.plot_analytic_numeric(four_2, finite_sol_2, num, 50, save = False)
    # plt.plot(300, hq1.get_s())
    # plt.animate(sol_2, hq1.get_s())
    # plt.animate(hq1.get_s())
    # plt.animate(four_1, s = hq1.get_s())
    # plt.animate(four_2)
    # plt.animate(sol_2)
    # plt.rmse_plot(rmse_2, s = hq1.get_s(), name='3', save = True)
    plt.rmse_plot(rmse_fourier_2, num, save = True)