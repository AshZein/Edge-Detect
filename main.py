import cv2
import numpy as np
import  image_filter as imf
import sys
import os


def open_img_grey(path: str)-> np.ndarray:
    img = cv2.imread(path)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grey_img

def create_gradient(img: np.ndarray)-> np.ndarray:
    gaussian_kernel = imf.create_gaussian_kernel(5, 14)

    sobel_kernels = imf.get_sobel_kernels()

    gaussian_filtered = imf.image_convolution(img, gaussian_kernel)

    sobel_x = imf.image_convolution(gaussian_filtered, sobel_kernels[0])
    sobel_y = imf.image_convolution(gaussian_filtered, sobel_kernels[1])

    gradient = imf.combine_sobel(sobel_x, sobel_y)

    return gradient


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("please enter a file path.")
    elif len(sys.argv) == 2:
        img = open_img_grey(sys.argv[1])
        gradient = create_gradient(img)

        filename = sys.argv[1].split("/")
        filename = filename[-1]

        cv2.imwrite(f"output_{filename}", gradient)
    elif len(sys.argv) == 3:
        img = open_img_grey(sys.argv[1])
        gradient = create_gradient(img)

        cv2.imwrite(f"{sys.argv[2]}.jpg", gradient)
    else:
        print("too many arguments")
