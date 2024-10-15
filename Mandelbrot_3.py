# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:25:37 2024

@author: 周昱
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Mandelbrot:
    def __init__(self, resolution, bounds, max_iterations):
        self.resolution = resolution  
        self.bounds = bounds  
        self.max_iterations = max_iterations 
        
        self.mandelbrot_set = None 
    
    def GenerateSet(self):
        width, height = self.resolution
        xmin, xmax, ymin, ymax = self.bounds
        x = np.linspace(xmin, xmax, width)  
        y = np.linspace(ymin, ymax, height) 
        mandelbrot_set = np.zeros((height, width)) 

        for i in range(width):
            for j in range(height):
                zx, zy = x[i], y[j]
                c = complex(zx, zy)
                z = 0
                iteration = 0
                
               
                while abs(z) <= 2 and iteration < self.max_iterations:
                    z = z*z + c
                    iteration += 1
                if iteration < self.max_iterations:
                    log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                    nu = np.log(log_zn / np.log(2)) / np.log(2)
                    iteration = iteration + 1 - nu
            
                mandelbrot_set[j, i] = iteration 
        
        self.mandelbrot_set = mandelbrot_set  
    
    
    def plot(self):
        if self.mandelbrot_set is None:
            raise ValueError("mandelbrot_set = None")
        
        colors = [(1, 1, 1), (1, 0, 0), (0.5, 0, 0.5), (0, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom", colors)
        plt.imshow(self.mandelbrot_set, cmap=cmap, extent=self.bounds)
        plt.title("Mandelbrot Set 3",fontsize=10)
        plt.xlabel("Re(c)",fontsize=10)
        plt.ylabel("Im(c)",fontsize=10)
        plt.savefig("Mandelbrot_3.pdf", format="pdf", bbox_inches='tight')

mandelbrot = Mandelbrot((800, 800), (-2.0, 1.0, -1.5, 1.5), 200)
mandelbrot.GenerateSet()
mandelbrot.plot()
