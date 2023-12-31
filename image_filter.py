import numpy as np
import math
import cv2

# 3x3 sobel kernels
sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)


def create_gaussian_kernel(size: int, sigma=1) -> np.ndarray:
    """
    Initializes a numpy array that represents a 2-dimensional gaussian kernel of
    <size> rows and <size> columns.

    Math source: https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

    :param size: The dimension of the size x size array. Should be an odd number
    :param sigma: The standard deviation
    :return: A size x size gaussian filter kernel
    """
    fltr = np.zeros((size, size))  # initialize (size X size) gaussian filter

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

    # normalize the filter
    fltr /= np.sum(fltr)

    return fltr


def get_sobel_kernels():
    return sobel_kernel_x, sobel_kernel_y

def combine_sobel(sobel_x: np.ndarray, sobel_y: np.ndarray):
    """
    Compute the gradient of the image G = (G_x^2 + G_y^2)^(1/2)
    :param sobel_x: the gradient in the x direction
    :param sobel_y: the gradient in the y direction
    :return: the gradient of the image
    """
    img_size = sobel_x.shape

    output = np.zeros(img_size)

    x = 0
    y = 0
    while x < img_size[0]:
        while y < img_size[1]:
            output[x, y] = (((sobel_x[x, y]) ** 2) + ((sobel_y[x, y]) ** 2)) ** (1/2)
            y += 1
        y = 0
        x += 1

    return output


def gradient_direction(sobel_x: np.ndarray, sobel_y: np.ndarray):
    """
    compute the gradient direction. using angle = arctan(G_y / G_x)

    :param sobel_x: image with the sobel x-direction kernel applied
    :param sobel_y: image with the sobel y-direction kernel applied
    :return: 2d numpy array of directions at each pixel
    """
    output_size = sobel_x.shape

    output = np.zeros(output_size)

    x = 0
    y = 0
    while x < output_size[0]:
        while y < output_size[1]:
            output[x, y] = math.atan(sobel_x[x, y] / sobel_y[x, y])

            y += 1
        y = 0
        x += 1

    return output
