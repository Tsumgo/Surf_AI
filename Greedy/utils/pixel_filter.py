import numpy as np

def pixel_filter(image):
    new_pixels = np.zeros(image.shape, dtype=np.uint8)

    new_pixels[(image == 0)] = 255 
    # new_pixels[(image == 29)] = 255 

    return new_pixels