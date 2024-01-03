import numpy as np
import math
import utilities as util
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


def combine_sobel(sobel_x: np.ndarray, sobel_y: np.ndarray) -> np.ndarray:
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


def gradient_direction(sobel_x: np.ndarray, sobel_y: np.ndarray) -> np.ndarray:
    """
    compute the gradient direction (in degrees). using angle = arctan(G_y / G_x)

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
            if sobel_y[x, y] == 0 and sobel_x[x, y] == 0:  # would've be undefined, but 0 if denominator non-zero
                angle = 0
            elif sobel_x[x, y] == 0:  # would be undefined
                angle = 90
            else:
                angle = math.atan(sobel_y[x, y] / sobel_x[x, y])
                angle = angle * (180/math.pi)  # Convert to degrees

            output[x, y] = angle
            y += 1
        y = 0
        x += 1

    return output


def round_gradient_direction(grad_dir: np.ndarray) -> np.ndarray:
    """
    Round gradient direction angles to 0, 45, or 135 degrees
    :param grad_dir: 2D numpy array of angles (in degrees) for an image gradient
    :return: 2D numpy array of rounded angles from grad_dir
    """
    grad_size = grad_dir.shape

    output = np.zeros(grad_size)

    for x in range(grad_size[0]):
        for y in range(grad_size[1]):
            angle = grad_dir[x, y]

            if angle > 0:
                if 0 < angle < 22.5:
                    out_angle = 0
                elif 22.5 <= angle < 67.5:
                    out_angle = 45
                elif 67.5 <= angle < 112.5:
                    out_angle = 90
                else:
                    out_angle = 135

            elif angle < 0:
                if -22.5 < angle < 0:
                    out_angle = 0
                elif -67.5 < angle <= -22.5:
                    out_angle = 135
                elif -112.5 < angle <= -67.5:
                    out_angle = 90
                else:
                    out_angle = 45

            else:
                out_angle = 0

            output[x, y] = out_angle

    return output


def non_max_suppression(gradient: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Apply the non-max suppression algorithm to the image gradient to create thinner edges

    :param gradient: numpy array of the image's gradient
    :param direction: numpy array of rounded edge direction of gradient
    :return: numpy array of a new image gradient
    """
    grad_shape = gradient.shape

    output = np.zeros(grad_shape)

    for x in range(grad_shape[0]):
        for y in range(grad_shape[1]):
            curr_pixel = gradient[x, y]
            curr_direction = direction[x, y]

            if curr_direction == 0:
                adj_1 = (x + 1, y)
                adj_2 = (x - 1, y)

            elif curr_direction == 45:
                adj_1 = (x + 1, y + 1)
                adj_2 = (x - 1, y - 1)

            elif curr_direction == 90:
                adj_1 = (x, y + 1)
                adj_2 = (x, y - 1)

            else:
                adj_1 = (x - 1, y + 1)
                adj_2 = (x + 1, y - 1)

            if 0 <= adj_1[0] < grad_shape[0] and 0 <= adj_1[1] < grad_shape[1]:
                adj_pixel = gradient[adj_1[0], adj_1[1]]
                if adj_pixel > curr_pixel:
                    output[x, y] = 0
                    curr_pixel = -1 # set curr_pixel to -1 to avoid comparison of other adjacent pixel

            if curr_pixel != -1 and 0 <= adj_2[0] < grad_shape[0] and 0 <= adj_2[1] < grad_shape[1]:
                adj_pixel = gradient[adj_2[0], adj_2[1]]
                if adj_pixel > curr_pixel:
                    output[x, y] = 0
                    curr_pixel = -1

            if curr_pixel != -1:  # adjacent pixels had equal intensities
                output[x, y] = curr_pixel

    return output


def calculate_thresholds_hyster()-> tuple(float, float):
    high = 0
    low = 0
    return high, low
