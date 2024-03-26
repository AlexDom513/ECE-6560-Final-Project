# Alex Domagala
# ECE 6560

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

print(os.getcwd())
print('djkfdjskjfdslk')

img = Image.open('./images/boats.bmp')
img = np.array(img)
plt.imshow(img, cmap='gray')
plt.show()


# def linearHeat(I):3

#   for ()