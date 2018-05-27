import matplotlib.pyplot as plt
import numpy as np


#DISPLAYDATA Display 2D data in a nice grid
#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid. It returns the figure handle h and the 
#   displayed array if requested.
def displayData(X, example_width=None):

    # Set example_width automatically if not passed in
    if example_width == None:
        example_width = (int)(np.round(np.sqrt(X.shape[1])))

    # Compute rows, cols
    m, n = X.shape
    example_height = (int)(n / example_width)

    # Compute number of items to display
    display_rows = (int)(np.floor(np.sqrt(m)))
    display_cols = (int)(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), \
                              pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            
            # Copy the patch

            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex]))
            offset_height = pad + j * (example_height + pad)
            offset_width = pad + i * (example_width + pad)
            display_array[offset_height : offset_height+example_height,\
                          offset_width : offset_width+example_width] = \
                          X[curr_ex].reshape((example_height,example_width)).T / max_val
            curr_ex = curr_ex + 1

        if curr_ex >= m:
            break
    
    # Display Image
    plt.imshow(display_array, cmap='gray', vmin = -1, vmax = 1)

    # Do not show axis
    plt.axis('off')