import cv2
import numpy as np


def unsharp_mask(img, blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)


def smoother_edges(img, first_blur_size, second_blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    img = cv2.GaussianBlur(img, first_blur_size, 0)
    return unsharp_mask(img, second_blur_size, imgWeight, gaussianWeight)


def close_image(img, size = (5,5)):
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def open_image(img, size = (5,5)):
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def shrink_rect(rect, scale = 0.8):
    center, (width, height), angle = rect
    width = width * scale
    height = height * scale
    rect = center, (width, height), angle
    return rect


def clahe(img, clip_limit = 2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(5,5))
    return clahe.apply(img)


def get_sobel(img, size = -1):
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,2,0,size)
    abs_sobel64f = np.absolute(sobelx64f)
    return np.uint8(abs_sobel64f)


img = cv2.imread("blobs4.jpg")
# save color copy for visualizing
imgc = img.copy()
# resize image to make the analytics easier (a form of filtering)
resize_times = 5
img = cv2.resize(img, None, img, fx = 1 / resize_times, fy = 1 / resize_times)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# use sobel operator to evaluate high frequencies
sobel = get_sobel(img)
# experimentally calculated function - needs refining
clip_limit = (-2.556) * np.sum(sobel)/(img.shape[0] * img.shape[1]) + 26.557

# don't apply clahe if there is enough high freq to find blobs
if(clip_limit < 1.0):
    clip_limit = 0.1
# limit clahe if there's not enough details - needs more tests
if(clip_limit > 8.0):
    clip_limit = 8

# apply clahe and unsharp mask to improve high frequencies as much as possible
img = clahe(img, clip_limit)
img = unsharp_mask(img)

# filter the image to ensure edge continuity and perform Canny
# (values selected experimentally, using trackbars)
img_blurred = (cv2.GaussianBlur(img.copy(), (2*2+1,2*2+1), 0))
canny = cv2.Canny(img_blurred, 35, 95)

# find first contours
_, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# prepare black image to draw contours
canvas = np.ones(img.shape, np.uint8)
for c in cnts:
    l = cv2.arcLength(c, False)
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h

    # filter "bad" contours (values selected experimentally)
    if l > 500:
        continue
    if l < 20:
        continue
    if aspect_ratio < 0.2:
        continue
    if aspect_ratio > 5:
        continue
    if l > 150 and (aspect_ratio > 10 or aspect_ratio < 0.1):
        continue
    # draw all the other contours
    cv2.drawContours(canvas, [c], -1, (255, 255, 255), 2)

# perform closing and blurring, to close the gaps
canvas = close_image(canvas, (7,7))
img_blurred = cv2.GaussianBlur(canvas, (8*2+1,8*2+1), 0)
# smooth the edges a bit to make sure canny will find continuous edges
img_blurred = smoother_edges(img_blurred, (9,9))
kernel = np.ones((3,3), np.uint8)
# erode to make sure separate blobs are not touching each other
eroded = cv2.erode(img_blurred, kernel)
# perform necessary thresholding before Canny
_, im_th = cv2.threshold(eroded, 50, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(im_th, 11, 33)

# find contours again. this time mostly the right ones
_, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# calculate the mean area of the contours' bounding rectangles
sum_area = 0
rect_list = []
for i,c in enumerate(cnts):
    rect = cv2.minAreaRect(c)
    _, (width, height), _ = rect
    area = width*height
    sum_area += area
    rect_list.append(rect)
mean_area = sum_area / len(cnts)

# choose only rectangles that fulfill requirement:
# area > mean_area*0.6
for rect in rect_list:
    _, (width, height), _ = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box * 5)
    area = width * height

    if(area > mean_area*0.6):
        # shrink the rectangles, since the shadows and reflections
        # make the resulting rectangle a bit bigger
        # the value was guessed - might need refinig
        rect = shrink_rect(rect, 0.8)
        box = cv2.boxPoints(rect)
        box = np.int0(box * resize_times)
        cv2.drawContours(imgc, [box], 0, (0,255,0),1)

# resize for visualizing purposes
imgc = cv2.resize(imgc, None, imgc, fx = 0.5, fy = 0.5)
cv2.imshow("imgc", imgc)
cv2.imwrite("result3.png", imgc)
cv2.waitKey(0)
