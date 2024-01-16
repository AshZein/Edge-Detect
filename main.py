import cv2
import numpy as np
import  image_filter as imf
import utilities as util
import sys
import os


def open_img_grey(path: str) -> np.ndarray:
    opened_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return opened_img

def perform_edge_detect(image: np.ndarray) -> np.ndarray:
    gaussian_kernel = imf.create_gaussian_kernel(5, 1.4)

    sobel_kernels = imf.get_sobel_kernels()

    gaussian_filtered = util.image_convolution(image, gaussian_kernel)

    sobel_x = util.image_convolution(gaussian_filtered, sobel_kernels[0])
    sobel_y = util.image_convolution(gaussian_filtered, sobel_kernels[1])

    gradient = imf.combine_sobel(sobel_x, sobel_y)
    grad_dir = imf.gradient_direction(sobel_x, sobel_y)

    rounded_grad_dir = imf.round_gradient_direction(grad_dir)

    non_max_sup = imf.non_max_suppression(gradient, rounded_grad_dir)

    return non_max_sup


if __name__ == '__main__':
    # command: python3 main.py FILE_PATH

    if len(sys.argv) == 2:
        grey_img = open_img_grey(sys.argv[1])

        output = perform_edge_detect(grey_img)

        filename = sys.argv[1].split("/")
        filename = filename[-1]

        cv2.imwrite(f"output_{filename}", output)
        print("---DONE---")

    elif len(sys.argv) == 1:
        img_num = input("please enter the number of the test image from the test_images directory: ")

        filename = "test_img_" + img_num + ".jpg"
        if filename in os.listdir("test_images"):
            grey_img = open_img_grey(f"test_images/{filename}")

            output = perform_edge_detect(grey_img)

            cv2.imwrite(f"output_{filename}", output)
            print("---DONE---")

    else:
        print("invalid command.\n python3 main.py FILE_PATH")


    # if len(sys.argv) > 2:
    #     img = open_img_grey(sys.argv[2])
    #
    #     output = perform_edge_detect(img)
    #
    #     if len(sys.argv) == 3:
    #         filename = sys.argv[2].split("/")
    #         filename = filename[-1]
    #
    #         cv2.imwrite(f"output_{filename}", output)
    #
    #     elif len(sys.argv) == 4:
    #         cv2.imwrite(f"{sys.argv[3]}.jpg", output)
    #
    # else:
    #     print("invalid command.\n python3 main.py OPERATION FILE_PATH OUTPUT_FILE_NAME")
