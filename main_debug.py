import cv2
import numpy as np
import trackbar as tb
import auxcv as aux
from matplotlib.pyplot import imshow, scatter, show, savefig


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
imgc = img.copy()
resize_times = 5
img = cv2.resize(img, None, img, fx = 1 / resize_times, fy = 1 / resize_times)
cv2.imshow("blobs", img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel = get_sobel(img)
print("sobel_sum: ", np.sum(sobel)/(img.shape[0] * img.shape[1]))
clip_limit = (-2.556) * np.sum(sobel)/(img.shape[0] * img.shape[1]) + 26.557
print("clip_limit: ", clip_limit)
cv2.waitKey(0)

if(clip_limit < 1.0):
    clip_limit = 0.1
if(clip_limit > 10.0):
    clip_limit = 10
img = clahe(img, clip_limit)
img = aux.unsharp_mask(img)
#canny = tb.CannyTrackbar(img, "Canny")
img_blurred = (cv2.GaussianBlur(img.copy(), (2*2+1,2*2+1), 0))
canny = cv2.Canny(img_blurred, 35, 95)
#img_blurred = cv2.GaussianBlur(img.copy(), (4*2+1,4*2+1), 0)
#canny = cv2.Canny(img_blurred, 11, 33)

# CONTOURS
_, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)

canvas = np.ones(img.shape, np.uint8)
#cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
for c in cnts:
    l = cv2.arcLength(c, False)
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h
    #print(l)
    #print(aspect_ratio)
    
    # if l > 500:
    #     cv2.drawContours(canvas, [c], -1, (0, 0, 255), 2)
    #     print("here: " + str(l))
    # if l < 20:
    #     cv2.drawContours(canvas, [c], -1, (0, 0, 255), 2)
    #     print("here: " + str(l))
    # if aspect_ratio < 0.2:
    #     cv2.drawContours(canvas, [c], -1, (255, 0, 0), 2)
    # if aspect_ratio > 5:
    #     cv2.drawContours(canvas, [c], -1, (255, 0, 0), 2)
    # if l > 150 and (aspect_ratio > 10 or aspect_ratio < 0.1):
    #     cv2.drawContours(canvas, [c], -1, (255, 255, 255), 2)
    
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
    cv2.drawContours(canvas, [c], -1, (255, 255, 255), 2)
    cv2.imshow("cnts", canvas)
    
cv2.waitKey(0)
cv2.imshow("contours1", canvas)
cv2.waitKey(0)

canvas = aux.close_image(canvas, (7,7))
img_blurred = cv2.GaussianBlur(canvas, (8*2+1,8*2+1), 0)
img_blurred = aux.smoother_edges(img_blurred, (9,9))
kernel = np.ones((3,3), np.uint8)
dilated = cv2.erode(img_blurred, kernel)
_, im_th = cv2.threshold(dilated, 50, 255, cv2.THRESH_BINARY)
cv2.imshow("contours1", im_th)
cv2.waitKey(0)
canny = cv2.Canny(im_th, 11, 33)
#canny = tb.CannyTrackbar(canvas, "canny")
cv2.imshow("canny", canny)
cv2.waitKey(0)

_, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

sum_area = 0
rect_list = []
for i,c in enumerate(cnts):
    rect = cv2.minAreaRect(c)
    _, (width, height), _ = rect
    area = width*height
    sum_area += area
    rect_list.append(rect)
mean_area = sum_area / len(cnts)
#print(mean_area)

for rect in rect_list:
    _, (width, height), _ = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box * 5)
    area = width * height
    
    if(area > mean_area*0.6):
        rect = shrink_rect(rect, 0.8)
        box = cv2.boxPoints(rect)
        box = np.int0(box * resize_times)
        cv2.drawContours(imgc, [box], 0, (0,255,0),1)

imgc = cv2.resize(imgc, None, imgc, fx = 0.5, fy = 0.5)
cv2.imshow("imgc", imgc)
cv2.waitKey(0)

# counter = 0
# # loop over the contours
# for c in cnts:
#     # compute the center of the contour
#     print(counter)
#     M = cv2.moments(c)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])

#     #draw the contour and center of the shape on the image
#     area = cv2.arcLength(c,True)
#     if area > 50:
#         #cv2.drawContours(img, [c], -1, (255, 255, 255), 1)
#         x,y,w,h = cv2.boundingRect(c)
#         aspect_ratio = float(w)/h
#         if aspect_ratio < 1.4 and aspect_ratio > 0.6:
#             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         #rect  = cv2.minAreaRect(c)
#         #box = cv2.boxPoints(rect)
#         #box = np.int0(box)
#         #cv2.drawContours(img,[box],0,(0,0,255),2)
#         counter = counter + 1
#     #cv2.waitKey(0)
# 	#cv2.circle(img, (cX, cY), 3, (255, 255, 255), -1)

cv2.imwrite("result1.png", imgc)
