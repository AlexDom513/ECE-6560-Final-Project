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
    self.I_min_MSE = img + noise
    self.I_orig = img
    self.timestep = timestep
    self.iterations = iterations
    self.skip = 10
    self.min_MSE = 1e10
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
    currMSE = np.mean(np.power(self.I_orig - self.I, 2))
    if (currMSE < self.min_MSE):
      self.min_MSE = currMSE
      self.I_min_MSE = self.I
    return currMSE

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
  
######################################################################
# Total Variation PDE
######################################################################
class Total_Variation(PDE):
  def __init__(self, img_path, timestep, iterations):
    super().__init__(img_path, timestep, iterations)
    self.E = 1

  def run(self):
    for iter in range(self.iterations):
      for i in range(1,self.I.shape[0]-1):
        for j in range(1,self.I.shape[1]-1):
          self.I[i,j] = self.I[i,j] + self.timestep * (((self.Ix(i,j)**2)*self.Iyy(i,j) - 2*self.Ix(i,j)*self.Iy(i,j)*self.Ixy(i,j) + 
                                                        (self.Iy(i,j)**2)*self.Ixx(i,j) + (self.E**2)*(self.Ixx(i,j) + self.Iyy(i,j))) / 
                                                      (((self.Ix(i,j)**2) + (self.Iy(i,j)**2) + (self.E**2))**(3/2)))
      self.MSE_arr.append(self.MSE())
      if (iter % self.skip == 0):
        print('TV iteration ' + str(iter) + ' / ' + str(iterations))

######################################################################
# Custom (Sigmoidal Penalty) PDE
######################################################################
class Custom(PDE):
  def __init__(self, img_path, timestep, iterations):
    super().__init__(img_path, timestep, iterations)
    self.E = 1
    self.lambda_const = 100
    self.beta_const = 10.3
    self.c_const = 20

  def gradient(self, Ix, Iy, power):
    Ix_squared = np.power(Ix, 2)
    Iy_squared = np.power(Iy, 2)
    return np.power(Ix_squared + Iy_squared + (self.E**2), power)
  
  def e_alpha(self, beta, Ix_input, Iy_input, c):
    currGradient = self.gradient(Ix_input, Iy_input, -.5)
    return np.exp((-1/beta) * (currGradient - c))
  
  def dDx_LIx(self, k, beta, c, Ix_input, Iy_input, Ixx_input, Ixy_input):
    e_alpha = self.e_alpha(beta,Ix_input,Iy_input,c)
    gradient_1 = self.gradient(Ix_input,Iy_input,-.5)
    gradient_2 = self.gradient(Ix_input,Iy_input,-1.5)
    return (k/beta) * ( Ixx_input*gradient_1 - Ix_input*gradient_2 * (Ix_input*Ixx_input + Iy_input*Ixy_input) * ( e_alpha / ((1+e_alpha)**2) ) +
                        Ix_input*gradient_1 * (-1/beta) * e_alpha * gradient_1 * (Ix_input*Ixx_input + Iy_input*Ixy_input) * ( ((1+e_alpha)**-2) + e_alpha * (-2) * ((1+e_alpha)**-3) ))

  def dDy_LIy(self, k, beta, c, Ix_input, Iy_input, Iyy_input, Ixy_input):
    e_alpha = self.e_alpha(beta,Ix_input,Iy_input,c)
    gradient_1 = self.gradient(Ix_input,Iy_input,-.5)
    gradient_2 = self.gradient(Ix_input,Iy_input,-1.5)
    return (k/beta) * ( Iyy_input*gradient_1 - Iy_input*gradient_2 * (Ix_input*Ixy_input + Iy_input*Iyy_input) * ( e_alpha / ((1+e_alpha)**2) ) +
                        Iy_input*gradient_1 * (-1/beta) * e_alpha * gradient_1 * (Ix_input*Ixy_input + Iy_input*Iyy_input) * ( ((1+e_alpha)**-2) + e_alpha * (-2) * ((1+e_alpha)**-3) ))

  def run(self):
    for iter in range(self.iterations):
      for i in range(1,self.I.shape[0]-1):
        for j in range(1,self.I.shape[1]-1):

          # compute partial derivatives
          Ix  = self.Ix(i,j)
          Iy  = self.Iy(i,j)
          Ixx = self.Ixx(i,j)
          Iyy = self.Iyy(i,j)
          Ixy = self.Ixy(i,j)

          # compute update
          self.I[i,j] = self.I[i,j] + self.timestep * (self.dDx_LIx(self.lambda_const,self.beta_const,self.c_const,Ix,Iy,Ixx,Ixy) + 
                                                       self.dDy_LIy(self.lambda_const,self.beta_const,self.c_const,Ix,Iy,Iyy,Ixy))
      self.MSE_arr.append(self.MSE())
      if (iter % self.skip == 0):
        print('Custom iteration ' + str(iter) + ' / ' + str(iterations))


if __name__ == "__main__":

  # shared parameters
  img_path = './images/boats.bmp'
  timestep = .01
  iterations = 100

  # linear heat
  linerHeat = Linear_Heat(img_path, timestep, iterations)
  linerHeat.run()

  # total variation
  tv = Total_Variation(img_path, timestep, iterations)
  tv.run()

  # custom (sigmoidal penalty)
  custom = Custom(img_path, timestep, iterations)
  custom.run()

  # MSE plots
  plt.title('MSE vs. Diffusion Iterations')
  plt.xlabel('Iteration Number')
  plt.ylabel('MSE')
  plt.plot(linerHeat.MSE_arr, label='Quadratic Penalty')
  plt.plot(tv.MSE_arr, label='Linear Penalty')
  plt.plot(custom.MSE_arr, label='Sigmoidal Penalty')
  plt.legend()
  plt.show()

  # obtain images with lowest MSE
  plt.imshow(linerHeat.I_min_MSE, cmap='gray')
  plt.show()

  plt.imshow(tv.I_min_MSE, cmap='gray')
  plt.show()

  plt.imshow(custom.I_min_MSE, cmap='gray')
  plt.show()