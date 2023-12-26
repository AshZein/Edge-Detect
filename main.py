import cv2
import numpy as np

def open_img_grey(path: str):
    img = cv2.imread(path)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grey_img

if __name__ == '__main__':
    pass
