import numpy as np
import scipy.io as sio
from random import randint
from displayData import displayData

def normalizeData(patches):

    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer

    # Remove DC (mean of images). 
    patches = patches - np.mean(patches)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3 * np.std(patches)
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1
    
    return patches


def sampleIMAGES():
    # sampleIMAGES
    # Returns 10000 patches for training

    patchsize = 8   # we'll use 8x8 patches
    numpatches = 10000

    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # column per patch, 10000 columns. 
    patches = np.zeros((patchsize*patchsize, numpatches))

    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Fill in the variable called "patches" using data 
    #  from IMAGES.  
    #  
    #  IMAGES is a 3D array containing 10 images
    #  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
    #  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
    #  it. (The contrast on these images look a bit off because they have
    #  been preprocessed using using "whitening."  See the lecture notes for
    #  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
    #  patch corresponding to the pixels in the block (21,21) to (30,30) of
    #  Image 1

    data = sio.loadmat('IMAGES.mat')
    IMAGES = data['IMAGES']

    for i in range(numpatches):
        imgIdx = randint(0,9  )
        rowIdx = randint(0,504)
        colIdx = randint(0,504)
        patches[:,i] = IMAGES[rowIdx:rowIdx+8,colIdx:colIdx+8,imgIdx].flatten()

    patches = normalizeData(patches)

    return patches