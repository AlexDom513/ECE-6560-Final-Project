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
# displays the results. Note that code at end of script can be
# uncommented to enable saving of figures and data.
######################################################################

from PIL import Image
import os
import sys
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

    # images
    self.I = img + noise
    self.I_min_MSE = np.copy(self.I)
    self.I_orig = img

    # shared parameters
    self.timestep = timestep
    self.iterations = iterations
    self.skip = 10

    # image quality data
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
      self.I_min_MSE = np.copy(self.I)
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
  def __init__(self, img_path, timestep, iterations, E):
    super().__init__(img_path, timestep, iterations)
    self.E = E

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
  def __init__(self, img_path, timestep, iterations, E, lambda_const, beta_const, c_const):
    super().__init__(img_path, timestep, iterations)
    self.E = E
    self.lambda_const = lambda_const
    self.beta_const = beta_const
    self.c_const = c_const

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
    return (k/beta) * ( (Ixx_input*gradient_1 - Ix_input*gradient_2 * (Ix_input*Ixx_input + Iy_input*Ixy_input)) * ( e_alpha / ((1+e_alpha)**2) ) +
                        Ix_input*gradient_1 * (-1/beta) * e_alpha * gradient_1 * (Ix_input*Ixx_input + Iy_input*Ixy_input) * ( ((1+e_alpha)**-2) + e_alpha * (-2) * ((1+e_alpha)**-3) ))

  def dDy_LIy(self, k, beta, c, Ix_input, Iy_input, Iyy_input, Ixy_input):
    e_alpha = self.e_alpha(beta,Ix_input,Iy_input,c)
    gradient_1 = self.gradient(Ix_input,Iy_input,-.5)
    gradient_2 = self.gradient(Ix_input,Iy_input,-1.5)
    return (k/beta) * ( (Iyy_input*gradient_1 - Iy_input*gradient_2 * (Ix_input*Ixy_input + Iy_input*Iyy_input)) * ( e_alpha / ((1+e_alpha)**2) ) +
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
  iterations = 300

  # prompt user
  print('Test 1 - Baseline Smoothing')
  print('Test 2 - Linear Smoothing')
  print('Test 3 - Aggressive Smoothing')
  print('Test 4 - Tempered Smoothing')
  try:
    test_num = int(input('Enter Test Option (1,2,3,4) '))
  except:
    print('Invalid Test Entered!')
    sys.exit()

  #################################
  # test 1
  #################################
  if (test_num == 1):
    
    ###### - Linear Heat - ######
    timestep = .05
    linerHeat = Linear_Heat(img_path, timestep, iterations)
    linerHeat.run()

    ##### - TV - ######
    timestep = .1
    E = 4
    tv = Total_Variation(img_path, timestep, iterations, E)
    tv.run()

    ###### - Custom - ######
    timestep = .1
    E = 4
    lambda_const = 1
    beta_const = 1
    c_const = 0
    custom = Custom(img_path, timestep, iterations, E, lambda_const, beta_const, c_const)
    custom.run()

  #################################
  # test 2
  #################################
  if (test_num == 2):

    ###### - Linear Heat - ######
    timestep = .05
    linerHeat = Linear_Heat(img_path, timestep, iterations)
    linerHeat.run()

    ###### - TV - ######
    E = 4
    timestep = .1
    tv = Total_Variation(img_path, timestep, iterations, E)
    tv.run()

    ###### - Custom - ######
    E = 4
    lambda_const = 87
    beta_const = 20
    c_const = 43
    custom = Custom(img_path, timestep, iterations, E, lambda_const, beta_const, c_const)
    custom.run()

  #################################
  # test 3
  #################################
  if (test_num == 3):

    ###### - Linear Heat - ######
    timestep = .05
    linerHeat = Linear_Heat(img_path, timestep, iterations)
    linerHeat.run()

    ###### - TV - ######
    E = 4
    timestep = .1
    tv = Total_Variation(img_path, timestep, iterations, E)
    tv.run()

    ###### - Custom - ######
    E = 4
    timestep = .1
    lambda_const = 400
    beta_const = 55
    c_const = 150
    custom = Custom(img_path, timestep, iterations, E, lambda_const, beta_const, c_const)
    custom.run()

  #################################
  # test 4
  #################################
  if (test_num == 4):

    ###### - Linear Heat - ######
    timestep = .05
    linerHeat = Linear_Heat(img_path, timestep, iterations)
    linerHeat.run()

    ###### - TV - ######
    E = 4
    timestep = .1
    tv = Total_Variation(img_path, timestep, iterations, E)
    tv.run()

    ###### - Custom - ######
    E = 4
    lambda_const = 50
    beta_const = 27
    c_const = 46
    custom = Custom(img_path, timestep, iterations, E, lambda_const, beta_const, c_const)
    custom.run()

  # MSE plots
  curr_dir = os.getcwd()
  folder_name = 'generated_images'

  plt.title('MSE vs. Diffusion Iterations')
  plt.xlabel('Iteration Number')
  plt.ylabel('MSE')
  plt.plot(linerHeat.MSE_arr, label='Quadratic Penalty')
  plt.plot(tv.MSE_arr, label='Linear Penalty')
  plt.plot(custom.MSE_arr, label='Sigmoidal Penalty')
  plt.legend()
  # file_name = 'MSE_test' + str(test_num) + '.png'
  # save_path = os.path.join(curr_dir,folder_name)
  # save_path = os.path.join(save_path,file_name)
  # plt.savefig(save_path)
  plt.show()

  # obtain images with lowest MSE and save
  plt.title('Linear Heat')
  plt.imshow(linerHeat.I_min_MSE, cmap='gray')
  print('Linear Heat minimum MSE: ' + str(linerHeat.min_MSE))
  # file_name = 'LinearHeat_test' + str(test_num) + '.png'
  # save_path = os.path.join(curr_dir,folder_name)
  # save_path = os.path.join(save_path,file_name)
  # plt.savefig(save_path)
  plt.show()

  plt.title('TV')
  plt.imshow(tv.I_min_MSE, cmap='gray')
  print('TV minimum MSE: ' + str(tv.min_MSE))
  # file_name = 'TV_test' + str(test_num) + '.png'
  # save_path = os.path.join(curr_dir,folder_name)
  # save_path = os.path.join(save_path,file_name)
  # plt.savefig(save_path)
  plt.show()

  plt.title('Custom')
  plt.imshow(custom.I_min_MSE, cmap='gray')
  print('Custom minimum MSE: ' + str(custom.min_MSE))
  # file_name = 'Custom_test' + str(test_num) + '.png'
  # save_path = os.path.join(curr_dir,folder_name)
  # save_path = os.path.join(save_path,file_name)
  # plt.savefig(save_path)
  plt.show()

  # write the minimum computed MSE to a file
  # file_name = 'MSE_test' + str(test_num) + 'summary' + '.txt'
  # save_path = os.path.join(curr_dir,folder_name)
  # save_path = os.path.join(save_path,file_name)
  # with open(save_path, 'w') as file:
  #   file.write('test' + str(test_num) + ' minimum MSE values:\n')
  #   file.write('linear heat: ' + str(linerHeat.min_MSE) + '\n')
  #   file.write('tv: ' + str(tv.min_MSE) + '\n')
  #   file.write('custom: ' + str(custom.min_MSE) + '\n')
