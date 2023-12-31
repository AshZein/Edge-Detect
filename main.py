import cv2
import numpy as np
import  image_filter as imf
import utilities as util
import sys
import os


def open_img_grey(path: str) -> np.ndarray:
    img = cv2.imread(path)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grey_img


def perform_edge_detect(img: np.ndarray) -> np.ndarray:
    gaussian_kernel = imf.create_gaussian_kernel(5, 14)

    sobel_kernels = imf.get_sobel_kernels()

    gaussian_filtered = util.image_convolution(img, gaussian_kernel)

    sobel_x = util.image_convolution(gaussian_filtered, sobel_kernels[0])
    sobel_y = util.image_convolution(gaussian_filtered, sobel_kernels[1])

    gradient = imf.combine_sobel(sobel_x, sobel_y)
    grad_dir = imf.gradient_direction(sobel_x, sobel_y)

    rounded_grad_dir = imf.round_gradient_direction(grad_dir)

    non_max_sup = imf.non_max_suppression(gradient, rounded_grad_dir)

    return non_max_sup


if __name__ == '__main__':
    # command: python3 main.py FILE_PATH OUTPUT_FILE_NAME

    if len(sys.argv) > 2:
        img = open_img_grey(sys.argv[2])

        output = perform_edge_detect(img)

        if len(sys.argv) == 3:
            filename = sys.argv[2].split("/")
            filename = filename[-1]

            cv2.imwrite(f"output_{filename}", output)

        elif len(sys.argv) == 4:
            cv2.imwrite(f"{sys.argv[3]}.jpg", output)

    else:
        print("invalid command.\n python3 main.py OPERATION FILE_PATH OUTPUT_FILE_NAME")
