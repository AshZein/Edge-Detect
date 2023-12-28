import time

import numpy as np

import image_filter as imf
import cv2

if __name__ == "__main__":
    img = cv2.imread("test_img_1.jpg")

    gauss_kernel = imf.create_gaussian_kernel(5, 15)
    sobel_kernels = imf.get_sobel_kernels()

    # multiprocessing
    time0 = time.time()
    gauss_filtered_img = imf.image_convolution_mp(img, gauss_kernel)

    # Sobel kernels
    sobel_x_filtered = imf.image_convolution_mp(gauss_filtered_img, sobel_kernels[0])
    sobel_y_filtered = imf.image_convolution_mp(gauss_filtered_img, sobel_kernels[1])

    # Gradient
    grad_multi = imf.combine_sobel(sobel_x_filtered, sobel_y_filtered)
    time1 = time.time()

    print("With multiprocessing: ", time1-time0)

    # None multiprocessing
    time0 = time.time()
    gauss_filtered_img = imf.image_convolution(img, gauss_kernel)

    # Sobel kernels
    sobel_x_filtered = imf.image_convolution(gauss_filtered_img, sobel_kernels[0])
    sobel_y_filtered = imf.image_convolution(gauss_filtered_img, sobel_kernels[1])

    # Gradient
    grad_not_multi = imf.combine_sobel(sobel_x_filtered, sobel_y_filtered)

    time1 = time.time()
    print("No multiprocessing: ", time1-time0)

    assert grad_not_multi.shape == grad_multi.shape
    assert np.array_equal(grad_not_multi, grad_multi)
