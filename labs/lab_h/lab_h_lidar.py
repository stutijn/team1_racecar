"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_h.py << [Modify with your own file name!]

Title: Lab H << [Modify with your own title]

Author: Kripa Sudhir << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: Drive on the right side of red cones and on the left side of blue cones.
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import time
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils
import math

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.


########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()


MIN_CONTOUR_AREA = 2000
MIN_CONTOUR_AREA_EX = 500
MAX_CONTOUR_AREA = 20000
BLUE = ((90, 100, 100), (120, 255, 255))  # The HSV range for the color blue
RED = ((170, 100, 100), (5, 255, 255))  # The HSV range for the color red
WHITE = ((80, 44, 255), (125, 63, 255)) # HSV range for white
CROP_FLOOR = ((200, 0), (rc.camera.get_height(), rc.camera.get_width()))

STATES = {1: "RIGHT", 2: "LEFT", 3: "SEARCH", 4: "FORWARD"}
global current_state
current_state = ""


# Declare any global variables here


########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle
    global last_state 
    global contour_r_area, contour_b_area, contour_area, contour_center
    global current_state, scan
    
    speed = 0
    angle = 0 
    contour_r_area = 0
    contour_b_area = 0
    contour_area = 0
    contour_center = 0
    last_state = ""
    print("STARTING! 🥳")
    # Remove 'pass' and write your source code for the start() function here

# def are_cones_to_sides():


def update_contour():
    global contour_center
    global contour_area
    global current_state
    global contour_r_area, contour_b_area, lrg_contour_w
    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
        
        # TODO Part 2: Search for line colors, and update the global variables
        # contour_center and contour_area with the largest contour found

        # Search for contours of the current color
        contours_b = rc_utils.find_contours(image, BLUE[0], BLUE[1])
        contours_r = rc_utils.find_contours(image, RED[0], RED[1])
        contours_w = rc_utils.find_contours(image, WHITE[0], WHITE[1])
        lrg_contour_w = rc_utils.get_largest_contour(contours_w, MIN_CONTOUR_AREA)
        # print(contours_b, contours_r)
        # filt_contours_b = [b for b in contours_b if cv.contourArea(b) < MAX_CONTOUR_AREA]
        # filt_contours_r = [r for r in contours_r if cv.contourArea(r) < MAX_CONTOUR_AREA]
        lrg_contour_b = rc_utils.get_largest_contour(contours_b, MIN_CONTOUR_AREA)
        lrg_contour_r = rc_utils.get_largest_contour(contours_r, MIN_CONTOUR_AREA)

        if lrg_contour_b is not None:
            contour_b_area = rc_utils.get_contour_area(lrg_contour_b)
            contour_b_center = rc_utils.get_contour_center(lrg_contour_b)
            rc_utils.draw_circle(image, contour_b_center)
        else:
            contour_b_area = 0
            contour_b_center = None    
        if lrg_contour_r is not None:
            contour_r_area = rc_utils.get_contour_area(lrg_contour_r)
            contour_r_center = rc_utils.get_contour_center(lrg_contour_r)
            rc_utils.draw_circle(image, contour_r_center)
        else:
            contour_r_area = 0
            contour_r_center = None
        
        if contour_r_area > contour_b_area:
            contour_center = rc_utils.get_contour_center(lrg_contour_r)
        elif contour_r_area < contour_b_area:
            contour_center = rc_utils.get_contour_center(lrg_contour_b)
        

        rc.display.show_color_image(image)
        
    
    # print(f"Red area: {contour_r_area}; Blue area: {contour_b_area}")
        # Display the image to the screen
        
def navigate_red():
    pass

def navigate_blue():
    pass

def clamp_cone_angle(angle_c):
    a = 0
    if angle_c > 90:
        a = angle_c - 360
    return a

def search(current_state):
    pass

def maintain_distance(distance_c, target, kp, current_state):
    angle = 0
    error = target - distance_c
    if distance_c < 115 and current_state == "LEFT":
        angle = kp * error
    elif distance_c < 115 and current_state == "RIGHT":
        angle = -(kp * error)
    elif distance_c > 50 and current_state == "FORWARD":
        angle = 0
    elif distance_c > 50 and current_state == "SEARCH":
        angle = lidar_cone_search(current_state)
        
    return angle


def lidar_cone_search(current_state):
    global angle
    scan = rc.lidar.get_samples()
    angle_c, distance_c = rc_utils.get_lidar_closest_point(scan, (260, 100))
    if current_state == "RIGHT":
        if 345 <= angle_c < 350:
            angle = 0
        elif 270 < angle_c < 345:
            angle = 1
        elif angle_c <= 315:
            angle = -1
    elif current_state == "LEFT":
        if 10 <= angle_c < 15:
            angle = 0
        elif 45 > angle_c > 15:
            angle = -1
        elif angle_c >= 45:
            angle = 1

    return angle

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter

def turn_right(r_area, b_area):
    return r_area > b_area and r_area < MAX_CONTOUR_AREA and r_area > MIN_CONTOUR_AREA

def turn_left(r_area, b_area):
    return r_area < b_area and b_area < MAX_CONTOUR_AREA and b_area > MIN_CONTOUR_AREA

def update_new():
    global speed
    global angle
    global contour_r_area, contour_b_area
    global current_state
    global scan
    global last_state
    
    kp = 0.075
    target = 50
    speed = 0.25

    update_contour()

    print(f"Red area: {contour_r_area}; Blue area: {contour_b_area}")
    
    if contour_r_area > contour_b_area or contour_b_area == 0:
        current_state = "RIGHT"
    elif contour_r_area < contour_b_area or contour_r_area == 0:
        current_state = "LEFT"
    elif contour_r_area < 100 and contour_b_area < 100:
        current_state = "SEARCH"
        print(current_state)
    
    scan = rc.lidar.get_samples()
    angle_c, distance_c = rc_utils.get_lidar_closest_point(scan, (260, 100))

    error = target - distance_c
    if distance_c < 115 and current_state == "LEFT":
        angle = kp * error
    elif distance_c < 115 and current_state == "RIGHT":
        angle = -(kp * error)
    elif distance_c > 50 and current_state == "SEARCH":
        if current_state == "RIGHT":
            if 345 <= angle_c < 350:
                angle = 0
            elif 270 < angle_c < 345:
                angle = 1
            elif angle_c <= 315:
                angle = -1
        elif current_state == "LEFT":
            if 10 <= angle_c < 15:
                angle = 0
            elif 45 > angle_c > 15:
                angle = -1
            elif angle_c >= 45:
                angle = 1

    angle = rc_utils.clamp(angle, -1, 1)
    
    # print(distance_c)
    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    pass
    # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update_new, update_slow)
    rc.go()
