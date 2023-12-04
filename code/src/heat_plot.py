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

    def __init__(self, x:np.ndarray, t:np.ndarray):
        self.x = x
        self.t = t

    def plot(self, y:np.ndarray, time:int, s:float, save:bool = False):
        """
        Plotting the solution to the heat equation using finite difference method for a given time.
        """
        # Checking if time is given
        if time is None:
            raise ValueError('Time must be specified')
        
        # Plotting for a given time
        plt.figure()
        plt.plot(self.x, y[:, time], 'r')
        plt.xlim(0, self.x.max())
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f'Solution to Heat equation with s = {s:.2f}')
        if save:
            plt.savefig(os.path.join('../figure', f'solution_time_{time}_s_{s:.2f}_.png'))
        plt.show()
    
    def rmse_plot(self, rmse:np.ndarray, s:float, name:str = None, save:bool = False):
        """
        Plotting the rmse for all time steps
        """
        # Plotting rmse
        plt.figure()
        plt.plot(self.t, rmse, 'r')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.title(f'RMSE for exact solution and numerical scheme w/ s = {s:.3f}')
        if save:
            plt.savefig(os.path.join('../figure', f'rmse_error_{name}.png'))
        plt.show()

    def plot_analytic_numeric(self, u:np.ndarray, ue:np.ndarray, s:float, time:int, save:bool = False):
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
        plt.plot(self.x, ue[:, time], 'blue', label = 'Analytic')
        plt.plot(self.x, u[:, time], 'r--', label = 'Numeric')
        plt.xlim(0, self.x.max())
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f'Heat equation s = {s:.3f} for time = {time}' + r'$\Delta t$')
        plt.legend()
        if save:
            plt.savefig(os.path.join('../figure', f'analytic_numeric_s_{s:.3f}_.png'))
        plt.show()

    
    def animate(self, y:np.ndarray, s:float = None, save:bool = False):
        """
        Creating animation of the solution to the heat equation using finite difference method  
        """
        # Creating figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.x.max()), ylim=(y.min() - 0.2, y.max() + 0.2))
        line, = ax.plot([], [], 'r', lw=2)
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'u(x, t)')
        if s is not None:
            ax.set_title(f'Solution to the heat equation with s = {s:.2f}')
        else:
            ax.set_title('Solution to the heat equation')

        # Initialization function
        def init():
            line.set_data([], [])
            return line,

        # Animation function
        def animate(i):
            line.set_data(self.x, y[:, i])
            return line,

        # Creating animation
        anim = FuncAnimation(fig, animate, init_func=init, frames=self.t.shape[0], interval=0.9, blit=True)
        if save:
            anim.save(os.path.join('../figure', 'heat_eq.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()

        


