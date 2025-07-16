
"""

Purpose: [navigate around cones] << [Write the purpose of the script here]

Expected Outcome: program your RACECAR to navigate through a "cone slalom" course,
which is where your vehicle must swerve between red and blue cones to reach the finish line.
Drive on the right side of RED cones
Drive on the left side of BLUE cones

Idea: create mask for both blue and red. select the larger mask. if contour is blue, change
angle accordingly. if red, as does. 
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils


########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 36

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

BLUE = ((90, 150, 100), (120, 255, 255))  # The HSV range for the color blue
RED = ((170,50,50),(10,255,255))  # The HSV range for the color red


# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

queue = [] # The queue of instructions
cone_color = "" # The current color of the cone

state = "searching"

########################################################################################
# Functions
########################################################################################
def update_contour():
    global contour_center
    global contour_area
    global colorset
    global cone_color

    image = rc.camera.get_color_image()

   # image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
    
    colorset = [((BLUE),"BLUE"),((RED),"RED")]

    if image is None:
        contour_center = None
        contour_area = 0
        cone_color = ""

    max_area = 0
    largest_contour = None
    largest_color = ""

    for hsv_range, color_name in colorset: #because colorset is a tuple, can access multiple parts
        contours = rc_utils.find_contours(image, hsv_range[0], hsv_range[1])
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
        if contour is not None:
            area = rc_utils.get_contour_area(contour)
            if area > max_area: #identifying the largest contour
                max_area = area
                largest_contour = contour
                largest_color = color_name

    if largest_contour is not None:
        contour_center = rc_utils.get_contour_center(largest_contour)
        contour_area = max_area
        cone_color = largest_color
        rc_utils.draw_contour(image, largest_contour)
    else:
        contour_center = None
        contour_area = 0
        cone_color = ""

    rc.display.show_color_image(image)


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    global state

    if contour_center is None:
        state = "searching"

    update_contour()

    if cone_color == "BLUE":
        state = "left"
    elif cone_color == "RED":
        state = "right"

   # adjust_distance()
    if state == "left":
        speed = 0.3
        angle = -0.5
    elif state == "right":
        speed = 0.3
        angle = 0.5
    else:
        print("no cone found?")

    rc.drive.set_speed_angle(speed, angle)


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the contour area and x-position
    if rc.camera.get_color_image() is None:
        # If no image is found, print all X's and don't display an image
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If an image is found but no contour is found, print all dashes
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area))

        # Otherwise, print a line of dashes with a | indicating the contour x-position
        else:
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
