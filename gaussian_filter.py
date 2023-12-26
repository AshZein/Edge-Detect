import numpy as np
import cv2

def image_convolution(kernel: np.ndarray, image: np.ndarray):

    pass

def create_gaussian_kernel(size: int, sigma=1) -> np.ndarray:
    """
    Initializes a numpy array that represents a 2-dimensional gaussian kernel of <size> rows and <size> columns.

    :param size: The dimension of the size x size array. Should be an odd number
    :param sigma: The standard deviation
    :return: A size x size gaussian filter kernel
    """
    fltr = np.zeros((size, size)) # initialize (size X size) gaussian filter

    i = 0
    j = 0
    while i < size:
        while j < size:
            x = (size // 2) - i
            y = (size // 2) - j

            exponent = (-1) * ((x ** 2 + y ** 2) / (2 * (sigma ** 2)))

            val = np.exp(exponent)
            val = val * (1 / (2 * np.pi * (sigma ** 2)))
            fltr[i, j] = val

            j += 1

        j = 0
        i += 1

    return fltr

if __name__ == "__main__":
    print(create_gaussian_kernel(5))