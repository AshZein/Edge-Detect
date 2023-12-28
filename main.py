import cv2
import numpy as np
import  image_filter as imf


def open_img_grey(path: str):
    img = cv2.imread(path)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grey_img


if __name__ == '__main__':
    img = open_img_grey("test_img_1.jpg")

    cv2.imwrite("original.jpg", img)

    # apply gaussian filter
    gauss_kernel = imf.create_gaussian_kernel(5, 15)
    gauss_filtered_img = imf.image_convolution(img, gauss_kernel)

    sobel_kernels = imf.get_sobel_kernels()
    sobel_x_filtered = imf.image_convolution(gauss_filtered_img, sobel_kernels[0])
    sobel_y_filtered = imf.image_convolution(gauss_filtered_img, sobel_kernels[1])

    cv2.imwrite("out_sobel_x.jpg", sobel_x_filtered)
    cv2.imwrite("out_sobel_y.jpg", sobel_y_filtered)

    combined_gradient = imf.combine_sobel(sobel_x_filtered, sobel_y_filtered)
    cv2.imwrite("out_sobel_combined.jpg", combined_gradient)
