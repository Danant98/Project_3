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


def fourier_1(x:np.ndarray, t:np.ndarray, n:int = 5):
    """
    Solution with Fourier series for the first example
    """
    u = np.zeros((x.shape[0], t.shape[0]))
    w = np.zeros((x.shape[0], t.shape[0]))
    for time in range(w.shape[1]):
        for j in range(w.shape[0]):
            score = 0
            for k in range(1, n):
                if k % 2 == 0:
                    nom = np.sin(t[time]) + ((k**2) * (np.pi**2) * np.cos(t[time]))
                    denom = k * np.pi * (1 + ((k**4) * (np.pi**4)))
                    score += (nom / denom) * np.sin(k * np.pi * x[j])
                elif k % 2 == 1:
                    # First sum
                    nom_1 = np.sin(t[time]) + (k**2 * np.pi**2 * np.cos(t[time]))
                    denom_1 = k * np.pi * (1 + k**4 * np.pi**4)
                    term_1 = (nom_1 / denom_1) * np.sin(k * np.pi * x[j])
                    # Second sum
                    coeff_term_1 = 8 / (k**3 * np.pi**3)
                    coeff_term_2 = 2 * ((k * np.pi) / (1 + k**4 * np.pi**4))
                    term_2 = (coeff_term_1 - coeff_term_2) * np.exp(-k**2 * np.pi**2 * t[time]) * np.sin(k * np.pi * x[j])
                    score += term_1 + term_2
            w[j, time] = score

    # Solving for u(x, t)
    for time in range(w.shape[1]):
        for j in range(w.shape[0]):
            u[j, time] = w[j, time] + (1 - x[j]) * np.sin(t[time])
    return u

def fourier_2(x:np.ndarray, t:np.ndarray, n:int = 3):
    """
    Solution with Fourier series for the second example
    """
    u = np.zeros((x.shape[0], t.shape[0]))
    for time in range(u.shape[1]):
        for j in range(u.shape[0]):
            score = 0
            for k in range(1, n):
                # First term in the coefficients
                nom_1 = 4 * ((2 * k + 1)**2 * np.pi**2 - 2)
                denom_1 = (2 * k + 1)**3 * np.pi**3 * ((2 * k + 1)**2 * np.pi**2 - 1)
                exp = np.exp(-(2 * k + 1)**2 * np.pi**2 * t[time])
                term_1 = (nom_1 / denom_1) * exp
                # Second term in the coefficients
                nom_2 = 4 * np.exp(-t[time])
                denom_2 = (2 * k + 1) * np.pi * ((2 * k + 1)**2 * np.pi**2 - 1)
                term_2 = nom_2 / denom_2
                # Combining terms
                score += (term_1 + term_2) * np.sin((2 * k + 1) * np.pi * x[j])
            u[j, time] = score
    return u

# Initializing numerical scheme
hq1 = hq()

# Gettiing the position and time arrays
x, t = hq1.get_x_t()

# Computing h(x, t)
h = H(lambda x, t: np.exp(-t), x, t)

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
f_1 = fourier_1(x, t)
f_2 = fourier_2(x, t)

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