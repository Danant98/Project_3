#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np

class heat_eq:

    def __init__(self, dt:float = 0.001, dx:float = 0.1, T:int = 5, l:int = 1):
        # Initializing time and position arrays
        self.x = np.arange(0, l + dx, dx)
        self.t = np.arange(0, T + dt, dt)
        self.s = dt / (dx**2)
        self.dt = dt
        # Initializing the solution array u(x, t)
        self.u = np.zeros((self.x.shape[0], self.t.shape[0]))
    
    def get_x_t(self):
        """
        Get the time and position arrays
        """
        return self.x, self.t
    
    def get_s(self):
        """
        Get the value of s
        """
        return self.s

    def finite_diff(self, f:np.ndarray, g:np.ndarray, phi:np.ndarray, rho:np.ndarray):
        """
        Solving the heat equation using finite difference method with given boundary and initial conditions
        """
        # Setting initial condition
        self.u[:, 0] = phi

        # Setting boundary conditions
        self.u[0, :] = f
        self.u[-1, :] = g

        # Solving the heat equation using finite difference method
        for n in range(1, self.u.shape[1] - 1):
            for i in range(1, self.u.shape[0] - 1):
                self.u[i, n] = self.s * (self.u[i + 1, n - 1] + self.u[i - 1, n - 1]) +\
                                   (1 - 2 * self.s) * self.u[i, n - 1] + self.dt * rho[i, n - 1]
        return self.u


if __name__ == '__main__':
    pass
