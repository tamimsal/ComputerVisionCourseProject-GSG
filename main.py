import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def sorted_directory_listing_with_os_listdir(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

def stritchImages():
    stitcher = cv2.Stitcher_create()
    files = sorted_directory_listing_with_os_listdir('uploads')
    leftImage = cv2.imread('uploads/' + files[0])
    count = 0
    for file in files[1:]:
        rightImage = cv2.imread('uploads/' + file)
        status, stitched_image = stitcher.stitch((leftImage, rightImage))  
  
        if status == cv2.Stitcher_OK:
            print("Stitching successful!")
            leftImage = stitched_image
            cv2.imwrite("uploads/result.jpg", stitched_image)
            count+=1
        else:
            print("Stitching failed!")
    print(count)




def cannyEdgeDetection():
    orginalImage = cv2.imread("uploads/result.jpg")
    grayImage = cv2.cvtColor(orginalImage, cv2.COLOR_BGR2GRAY)
    median_value = np.median(grayImage)
    lower_threshold = int(max(0, 0.7 * median_value))
    upper_threshold = int(min(255, 1.3 * median_value))
    canny_edges = cv2.Canny(grayImage, lower_threshold, upper_threshold)
    cv2.imwrite("uploads/cannyResult.jpg", canny_edges)





stritchImages()
cannyEdgeDetection()








