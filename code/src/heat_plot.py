#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn; sn.set_style('darkgrid')
import os
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error

class Plot:

    def __init__(self, x:np.ndarray, t:np.ndarray, y:np.ndarray):
        self.x = x
        self.y = y
        self.t = t

    def plot(self, time:int, s:float, save:bool = False):
        """
        Plotting the solution to the heat equation using finite difference method for a given time.
        """
        # Checking if time is given
        if time is None:
            raise ValueError('Time must be specified')
        
        # Plotting for a given time
        plt.figure()
        plt.plot(self.x, self.y[:, time], 'r')
        plt.xlim(0, self.x.max())
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f'Solution to Heat equation with s = {s:.2f}')
        if save:
            plt.savefig(os.path.join('../figure', f'solution_time_{time}_s_{s:.2f}_.png'))
        plt.show()
    
    def rmse_plot(self, u:np.ndarray, uhat:np.ndarray, label:str = None, save:bool = False):
        """
        Plotting the rmse for all time steps
        """
        # Computing the rmse error
        rmse = np.zeros((u.shape[1]))
        for time in range(u.shape[1]):
            rmse[time] = mean_squared_error(u[:, time], uhat[:, time])
        
        # Plotting rmse
        plt.figure()
        plt.plot(range(rmse.shape[0]), rmse, 'r')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.title('RMSE for Fourer solution and numerical solution')
        if save:
            plt.savefig(os.path.join('../figure', f'rmse_error_{label}.png'))
        plt.show()


    def plot_analytic_numeric(self, ue:np.ndarray, s:float, time:int, save:bool = False):
        """
        Plotting analytcal and numerical solutions
        """
        if time is None:
            raise ValueError('Time must be specified')
        elif time == 0:
            raise Exception('Initial value is always plotted, please specify time value > 0')

        # Plotting for a given time
        plt.figure()
        plt.plot(self.x, ue[:, 0], 'black', label = 'Initial condition')
        plt.plot(self.x, ue[:, time], 'b--', label = 'Analytic')
        plt.plot(self.x, self.y[:, time], 'r--', label = 'Numeric')
        plt.xlim(0, self.x.max())
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f'Heat equation s = {s} for time = {time}' + r'$\Delta t$')
        if save:
            plt.savefig(os.path.join('../figure', 'analytic_numeric.png'))
        plt.show()

    
    def animate(self, s:float, save:bool = False):
        """
        Creating animation of the solution to the heat equation using finite difference method  
        """
        # Creating figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.x.max()), ylim=(-0.5, self.y.max() + 0.2))
        line, = ax.plot([], [], 'r', lw=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f'Solution to the heat equation with s = {s:.2f}')

        # Initialization function
        def init():
            line.set_data([], [])
            return line,

        # Animation function
        def animate(i):
            line.set_data(self.x, self.y[:, i])
            return line,

        # Creating animation
        anim = FuncAnimation(fig, animate, init_func=init, frames=self.t.shape[0], interval=20, blit=True)
        if save:
            anim.save(os.path.join('../figure', 'heat_eq.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()

        


