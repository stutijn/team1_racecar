"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: racecar_vision.py

Title: RACECAR Vision

Author: Kripa Sudhir

Purpose: To test different features of OpenCV and the Camera class in RACECAR
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here


########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass

def process_image():
    # Get a frame from camera and store it
    image = rc.camera.get_color_image()
    
    # Crop the image
    image = rc_utils.crop(image, (180, 0), (rc.camera.get_height(), rc.camera.get_width()))

    hsv_lower = (10, 50, 50)
    hsv_upper = (20, 255, 255)
    
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    mask = cv.inRange(image, hsv_lower, hsv_upper)

    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    
    CONTOUR_MIN = 100
    
    filt_cont = []
    max_cont = contours[0]
    for x in contours:
        if cv.contourArea(x) > CONTOUR_MIN:
            filt_cont.append(x)
            if cv.contourArea(x) > cv.contourArea(max_cont):
                max_cont = x


    cv.drawContours(image, [max_cont], 0, (0, 255, 0), 3)

    center = rc_utils.get_contour_center(max_cont)

    circle = cv.circle(image, (center[1], center[0]), 6, (0, 255, 255), -1)

    # rc.display.show_color_image(mask)

    # dimension = image.shape # stores the image height, width, and channel numbers (depth)
    # # in a tuple
    # print(f"\n==================================")
    # print(f"Height of Image (rows): {dimension[0]}")
    # print(f"Width of Image (cols): {dimension[1]}")
    # print(f"Depth of Image (channels): {dimension[2]}")
    # print(f"\n==================================")

    # # Finds pixel in the middle of the screen
    # row = dimension[0] // 2
    # col = dimension[1] // 2

    # # Extracts and prints blue, green, and red values
    # blue = image[row][col][0]
    # green = image[row][col][1]
    # red = image[row][col][2]
    # print(f"BGR: ({blue}, {green}, {red})")

    # #Display color to screen
    # BGR_color = (blue, green, red)
    # BGR_image = np.zeros((300, 300, 3), np.uint8)
    # BGR_image[:] = BGR_color
    # cv.namedWindow('BGR Color Display', cv.WINDOW_NORMAL)
    # cv.imshow('BGR Color Display', BGR_image)



    # Display the frame to the screen
    rc.display.show_color_image(image)
    
# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    process_image()

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
