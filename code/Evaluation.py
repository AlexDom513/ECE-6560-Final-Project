######################################################################
# evaluation.py
######################################################################
# Alexander Domagala
# ECE 6560 - Final Project - Image Smoothing
# Spring 2024
######################################################################
# Aggregates the functionality contained in the following notebooks:
# Linear_Heat.ipynb, TV.ipynb, and Sigmoid.ipynb.
# Runs the different PDE schemes on the test image and
# displays the results.
######################################################################

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

######################################################################
# PDE
######################################################################
class PDE:
  def __init__(self, img_path, timestep, iterations):
    start_pix = 300
    end_pix =start_pix+100

    img = Image.open(os.path.join(os.getcwd(),img_path))
    img = np.array(img)
    img = img[start_pix:end_pix,start_pix:end_pix]

    mean = 0
    std_dev = 20
    noise = np.random.normal(mean, std_dev, (img.shape[0],img.shape[1]))

    self.I = img + noise
    self.I_orig = img
    self.timestep = timestep
    self.iterations = iterations
    self.skip = 10
    self.MSE_arr = []

  def Ix(self,i,j):
    return (self.I[i+1,j] - self.I[i-1,j]) / 2

  def Iy(self,i,j):
    return (self.I[i,j+1] - self.I[i,j-1]) / 2

  def Ixx(self,i,j):
    return (self.I[i+1,j] - 2*self.I[i,j] + self.I[i-1,j])

  def Iyy(self,i,j):
    return (self.I[i,j+1] - 2*self.I[i,j] + self.I[i,j-1])

  def Ixy(self,i,j):
    return (self.I[i+1,j+1] - self.I[i+1,j-1] - self.I[i-1,j+1] + self.I[i-1,j-1]) / 4
  
  def MSE(self):
    return np.mean(np.power(self.I_orig - self.I, 2))

######################################################################
# Linear Heat PDE
######################################################################
class Linear_Heat(PDE):
  def __init__(self, img, timestep, iterations):
    super().__init__(img, timestep, iterations)

  def run(self):
    for iter in range(self.iterations):
      for i in range(1,self.I.shape[0]-1):
        for j in range(1,self.I.shape[1]-1):
          self.I[i,j] = self.I[i,j] + self.timestep * (self.Ixx(i,j) + (self.Iyy(i,j)))
      self.MSE_arr.append(self.MSE())
      if (iter % self.skip == 0):
        print('Linear Heat iteration ' + str(iter) + ' / ' + str(iterations))
    print('Linear Heat complete!')
    return self.MSE_arr
  







if __name__ == "__main__":

  # linear heat equation
  img_path = './images/boats.bmp'
  timestep = .01
  iterations = 100
  linerHeat = Linear_Heat(img_path, timestep, iterations)
  MSE = linerHeat.run()

  plt.plot(MSE)
  plt.show()