import numpy as np
import math
import cv2

# 3x3 sobel kernels
sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)

def apply_kernel(image: np.ndarray, image_pos: tuple, kernel: np.ndarray):
    """
    Helper function for image_convolution. Applies the kernel to the image at <image_pos>.

    NOTE: if the bound of the image is exceeded the value for one of the terms
    of the pixel value will
          be set to 0

    Math Source: https://homepages.inf.ed.ac.uk/rbf/HIPR2/convolve.htm

    :param image: image to apply convolution
    :param image_pos: current pixel to apply kernel
    :param kernel: the gaussian filter kernel
    :return: the pixel value
    """
    output = 0

    kernel_size = kernel.shape
    image_size = image.shape

    # apply each kernel value
    m = 0  # row
    n = 0  # column
    while m < kernel_size[0]:
        while n < kernel_size[1]:
            if image_pos[0] + m >= image_size[0] or image_pos[1] + n >= image_size[1]:
                # exceeded image boundary
                val = 0
            else:
                val = image[image_pos[0] + m, image_pos[1] + n] * kernel[m, n]

            output += val

            n += 1
        m += 1
        n = 0

    return output


def image_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img_size = image.shape  # tuple of image dimension

    output = np.zeros(img_size)  # output image array

    # apply image kernel to each image pixel (x,y)
    x = 0
    y = 0
    while x < img_size[0]:
        while y < img_size[1]:
            output[x, y] = apply_kernel(image, (x, y), kernel)
            y += 1
        y = 0
        x += 1

    # Round output to integer
    output = np.round(output).astype(int)

    return output


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


if __name__ == "__main__":
    kern = create_gaussian_kernel(5)
    print(kern)
    print(kern.shape)
