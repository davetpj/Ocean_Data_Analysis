### SVD script for Week 13 (image process) 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['PROJ_LIB'] = 'D:\Park\Python\Library\share\proj'

####################### Import Data
im_orig = Image.open('marinesnow3.jpg')
print(np.array(im_orig).shape)
#plt.imshow(im_orig)
 
## converting the color image to black/white   
im_bw = im_orig.convert('L')
im = np.array(im_bw)

print(im.shape)

############### Singular Value Decomposition (SVD)
U, s, V = np.linalg.svd(im)

## converting Eigenvalues to a diagonal matrix 
sm = np.zeros([len(U), len(V)])
for j in range(0, len(U)):
    sm[j,j] = s[j]

im_recon = np.matmul( np.matmul(U,sm), V )

plt.imshow(im_recon, cmap='gray')
           
############### Clipping out Primary Modes
## 직접 코딩해보세요 


