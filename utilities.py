import numpy as np


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