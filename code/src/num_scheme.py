#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
import os
import matplotlib.pyplot as plt


class heat_eq:

    def __init__(self, k:float = 0.05, h:float = 0.1, T:int = 5, l:int = 1):
        # Initializing time and position arrays
        self.x = np.arange(0, l + h, h)
        self.t = np.arange(0, T + k, k)
        self.s = k / h**2
        self.k = k
        # Initializing the solution array u(x, t)
        self.u = np.zeros((self.x.shape[0], self.x.shape[0])) 
    
    def get_x_t(self):
        """
        Get the time and position arrays
        """
        return self.x, self.t

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
    
    def animate(self, save:bool = False):
        """
        Creating animation of the solution to the heat equation using finite difference method  
        """
        # Creating figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.x.shape[0]), ylim=(0, self.u.max() + 1))
        line, = ax.plot([], [], lw=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f'Solution to the heat equation with s = {self.s}')

        # Initialization function
        def init():
            line.set_data([], [])
            return line,

        # Animation function
        def animate(i):
            line.set_data(self.x, self.u[:, i])
            return line,

        # Creating animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, init_func=init, frames=self.t.shape[0], interval=20, blit=True)
        if save:
            anim.save(os.path.join('../figure', 'heat_eq.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()


    def plot(self, time:int, save:bool = False):
        """
        Plotting the solution to the heat equation using finite difference method for a given time.
        """
        # Checking if time is within bounds and given
        if time > self.t.shape[0]:
            raise ValueError('Time is out of bounds')
        if time == None:
            raise ValueError('Time is not given')
        
        # Plotting for a given time
        plt.plot(self.x, self.u[:, time])
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f'Solution to Heat equation with s = {self.s}')
        if save:
            plt.savefig(os.path.join('../figure', f'solution_time_{time}_s_{self.s}_.png'))
        plt.show()
        




if __name__ == '__main__':
    pass
