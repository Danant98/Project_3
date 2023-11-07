#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
import os
import matplotlib.pyplot as plt


class heat_eq:

    def __init__(self, k:float = 0.1, h:float = 0.5, T:int = 10, l:int = 5):
        # Initializing time and position arrays
        self.x = np.arange(0, l + h, h)
        self.t = np.arange(0, T + k, k)
        self.s = k / h**2
        self.k = k
        # Initializing the solution array u(x, t)
        self.u = np.zeros((self.x.shape[0], self.x.shape[0])) 
    
    def get_x_t_s(self):
        """
        Get the time and position arrays along with the s value
        """
        return self.x, self.t, self.s

    def finite_diff(self, f:np.ndarray, g:np.ndarray, phi:np.ndarray, rho:np.ndarray):
        """
        Solving the heat equation using finite difference method with given boundary and initial conditions
        """
        # Setting boundary conditions
        self.u[0, :] = f
        self.u[-1, :] = g
        # Setting initial condition
        self.u[:, 0] = phi

        # Solving the heat equation using finite difference method
        for n in range(self.u.shape[1]):
            for i in range(1, self.u.shape[0] - 1):
                self.u[i, n + 1] = self.s * (self.u[i + 1, n] + self.u[i - 1, n]) +\
                                   (1 - 2 * self.s) * self.u[i, n] + self.k * rho[i, n]
        return self.u






if __name__ == '__main__':
    pass
