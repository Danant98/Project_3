#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np

# Creating object for analytical solution
class Analytic:

    def __init__(self, x:np.ndarray, t:np.ndarray, l:float):
        self.x = x
        self.t = t
        self.l = l
    
    def solve(self, h:np.ndarray, g:np.ndarray, f:np.ndarray):
        """
        Analyticla solution to the heat equation
        """
        ue = np.zeros((self.x.shape[0], self.t.shape[0]))
        for n in range(ue.shape[1]):
            par = self.x * (g[n] / h[-1, n]) + ((self.l - self.x) * (f[n] / h[0, n]))
            ue[:, n] = h[:, n] * par
        return (1 / self.l) * ue







