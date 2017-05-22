"""
usage: threshold_custom = tb.SimpleTrackbar(img, "ImgThresh")
"""
import cv2
import numpy as np


def empty_function(*arg):
    pass


def SimpleTrackbar(img, win_name):
    trackbar_name = win_name + "Trackbar"

    cv2.namedWindow(win_name)
    cv2.createTrackbar(trackbar_name, win_name, 0, 255, empty_function)

    while True:
        trackbar_pos = cv2.getTrackbarPos(trackbar_name, win_name)
        _, img_th = cv2.threshold(img, trackbar_pos, 255, cv2.THRESH_BINARY)
        cv2.imshow(win_name, img_th)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return trackbar_pos


def CannyTrackbar(img, win_name):
    trackbar_name = win_name + "Trackbar"

    cv2.namedWindow(win_name)
    cv2.resizeWindow(win_name, 500,100)
    cv2.createTrackbar("1", win_name, 0, 255, empty_function)
    cv2.createTrackbar("2", win_name, 0, 255, empty_function)
    cv2.createTrackbar("3", win_name, 0, 255, empty_function)
    cv2.createTrackbar("4", win_name, 0, 255, empty_function)

    while True:
        trackbar_pos1 = cv2.getTrackbarPos("1", win_name)
        trackbar_pos2 = cv2.getTrackbarPos("2", win_name)
        trackbar_pos3 = cv2.getTrackbarPos("3", win_name)
        trackbar_pos4 = cv2.getTrackbarPos("4", win_name)
        img_blurred = cv2.GaussianBlur(img.copy(), (trackbar_pos3*2+1,trackbar_pos3*2+1), trackbar_pos4)
        canny = cv2.Canny(img_blurred, trackbar_pos1, trackbar_pos2)
        cv2.imshow(win_name, canny)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            break

    cv2.destroyAllWindows()
    return canny


def HoughTrackbar(canny, img_org, win_name):
    trackbar_name = win_name + "Trackbar"

    cv2.namedWindow(win_name)
    cv2.resizeWindow(win_name, 500,100)
    cv2.createTrackbar("1", win_name, 1, 255, empty_function)
    cv2.createTrackbar("2", win_name, 1, 255, empty_function)
    cv2.createTrackbar("3", win_name, 1, 255, empty_function)
    cv2.createTrackbar("4", win_name, 1, 255, empty_function)
    cv2.createTrackbar("5", win_name, 1, 255, empty_function)
    cv2.createTrackbar("6", win_name, 40, 255, empty_function)
    cv2.createTrackbar("7", win_name, 50, 255, empty_function)

    while True:
        img_org_copy = img_org.copy()
        trackbar_pos1 = cv2.getTrackbarPos("1", win_name)
        trackbar_pos2 = cv2.getTrackbarPos("2", win_name)
        trackbar_pos3 = cv2.getTrackbarPos("3", win_name)
        trackbar_pos4 = cv2.getTrackbarPos("4", win_name)
        trackbar_pos5 = cv2.getTrackbarPos("5", win_name)
        trackbar_pos6 = cv2.getTrackbarPos("6", win_name)
        trackbar_pos7 = cv2.getTrackbarPos("7", win_name)
        circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, trackbar_pos1+1, trackbar_pos2+1, trackbar_pos3+1, trackbar_pos4+1, trackbar_pos5+1, trackbar_pos6+1, trackbar_pos7+1)
        key = cv2.waitKey(0) & 0xFF
        if circles == None:
            img_org = img_org_copy
            continue
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img_org,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img_org,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('detected circles',img_org)
        if key == ord("c"):
            break
        img_org = img_org_copy
    
    return circles
