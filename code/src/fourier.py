#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and moduels 
import numpy as np

# Creating class for containing the Fourier solutions
class Fourier:

    def fourier_1(x:np.ndarray, t:np.ndarray, n:int = 2):
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
        return u, n

    # Second Fourier example
    def fourier_2(x:np.ndarray, t:np.ndarray, n:int = 10):
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
        return u, n


