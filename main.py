import cv2
import numpy as np
import  image_filter as imf
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
    images = os.listdir("test_images")
    images.remove(".DS_Store")
    for test_img in images:
        img = open_img_grey(f"test_images/{test_img}")
        gradient = create_gradient(img)

        cv2.imwrite(f"gradient_{test_img}.jpg", gradient)


    


