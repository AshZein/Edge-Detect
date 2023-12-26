import numpy as np
import cv2

def apply_kernel(image:np.ndarray, image_pos: tuple, kernel: np.ndarray):
    """
    Helper function for image_convolution. Applies the kernel to the image

    :param image: image to apply convolution
    :param image_pos: current pixel to apply kernel
    :param kernel: the gaussian filter kernel
    :return: the pixel value
    """
    output = 0

    kernel_size = kernel.shape
    image_size = image.shape

    m = 0
    n = 0
    while m < kernel_size[0]:
        while n < kernel_size[1]:
            if image_pos[0] + m >= image_size[0] or image_pos[1] + n >= image_size[1]:
                val = 0
            else:
                val = image[image_pos[0] + m, image_pos[1] + n] * kernel[m, n]

            output += val

            n += 1
        m += 1
        n = 0

    return output

def image_convolution(kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
    i = 0
    j = 0
    img_size = image.shape # tuple of image dimension

    output = np.zeros(img_size) # output image array

    while i < img_size[0]:
        while j < img_size[1]:
            output[i,j] = apply_kernel(image, (i, j), kernel)
            j += 1

        i += 1

    # MAYBE ROUND THE OUTPUT

    return output

def create_gaussian_kernel(size: int, sigma=1) -> np.ndarray:
    """
    Initializes a numpy array that represents a 2-dimensional gaussian kernel of <size> rows and <size> columns.

    :param size: The dimension of the size x size array. Should be an odd number
    :param sigma: The standard deviation
    :param factor: the factor to make kernel values integer
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

    # normalize the filter
    fltr /= np.sum(fltr)

    return fltr

if __name__ == "__main__":
    kern = create_gaussian_kernel(5)
    print(kern)
    print(kern.shape)