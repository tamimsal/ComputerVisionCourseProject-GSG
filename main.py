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
    for file in files[1:]:
        
        rightImage = cv2.imread('uploads/' + file)
        status, stitched_image = stitcher.stitch((leftImage, rightImage))    
        if status == cv2.Stitcher_OK:
            print("Stitching successful!")
        else:
            print("Stitching failed!")
    cv2.imwrite("uploads/result.jpg", stitched_image)


stritchImages()









