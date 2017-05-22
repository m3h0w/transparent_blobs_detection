import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_folder_avi(PATH = 'videos/'):
    print(len(glob.glob(os.path.join(PATH, '*.avi'))))
    FILE_NAMES = glob.glob(os.path.join(PATH, '*.avi'))
    return FILE_NAMES

def create_windows(NAMES: list = ["preview"]):
    cv2.startWindowThread()
    for NAME in NAMES:
        cv2.namedWindow(NAME)

def show_image(window_name, image, wait = False, new_window = True):
    if new_window:
        cv2.startWindowThread()
        cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)

def show_hist(img):
    plt.hist(img.ravel(),256,[0,256]); plt.show()

def image_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def close_image(img, size = (5,5)):
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def open_image(img, size = (5,5)):
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def unsharp_mask(img, blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)

def smoother_edges(img, first_blur_size, second_blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    img = cv2.GaussianBlur(img, first_blur_size, 0)
    return unsharp_mask(img, second_blur_size, imgWeight, gaussianWeight)

def compare_histograms_normal_and_masked(normal_image, masked_image):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.hist(normal_image.ravel(), 256, [0,256])
    ax2 = fig.add_subplot(212)
    plt.hist(masked_image[np.nonzero(masked_image)].ravel(), 256, [0,256])
    plt.show()