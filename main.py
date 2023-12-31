import cv2
import numpy as np
import  image_filter as imf
import utilities as util
import sys
import os


def open_img_grey(path: str)-> np.ndarray:
    img = cv2.imread(path)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grey_img

def create_gradient(img: np.ndarray)-> np.ndarray:
    gaussian_kernel = imf.create_gaussian_kernel(5, 14)

    sobel_kernels = imf.get_sobel_kernels()

    gaussian_filtered = util.image_convolution(img, gaussian_kernel)

    sobel_x = util.image_convolution(gaussian_filtered, sobel_kernels[0])
    sobel_y = util.image_convolution(gaussian_filtered, sobel_kernels[1])

    gradient = imf.combine_sobel(sobel_x, sobel_y)

    return gradient


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("please enter a file path.")
    elif len(sys.argv) == 2:
        img = open_img_grey(sys.argv[1])
        grad = create_gradient(img)

        filename = sys.argv[1].split("/")
        filename = filename[-1]

        cv2.imwrite(f"output_{filename}", grad)

    elif len(sys.argv) == 3:
        img = open_img_grey(sys.argv[1])
        grad = create_gradient(img)

        cv2.imwrite(f"{sys.argv[2]}.jpg", grad)
        
    else:
        print("too many arguments")
