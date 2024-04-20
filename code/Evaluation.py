######################################################################
# evaluation.py
######################################################################
# Alexander Domagala
# ECE 6560 - Final Project - Image Smoothing
# Spring 2024
######################################################################
# Aggregates the functions contained in the following notebooks:
# Linear_Heat.ipynb, TV.ipynb, and Sigmoid.ipynb.
# Runs the different PDE schemes on the test image and
# displays the results.
######################################################################

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Linear_Heat_PDE:
  def __init__(self, Img, timestep, iterations):
    self.I = Img
    self.I_orig = Img
    self.timestep = timestep
    self.iterations = iterations

  def run(self):

    MSE_arr = []
    for iter in range(self.iterations):
      for i in range(1,self.I.shape[0]-1):
        for j in range(1,self.I.shape[1]-1):
          self.I[i,j] = self.I[i,j] + self.timestep * ((self.I[i+1,j] - 2*self.I[i,j] + self.I[i-1,j]) + 
                                                       (self.I[i,j+1] - 2*self.I[i,j] + self.I[i,j-1]))
      # compute MSE after each iteration
      curr_MSE = np.mean(np.power(self.I_orig - self.I, 2))
      MSE_arr.append(curr_MSE)
    return MSE_arr
  
  

  