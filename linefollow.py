"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: linefollow.py

Title: Lab F - Line Follower

Author: TEAM 1

Purpose: Write a script to enable fully autonomous behavior from the RACECAR. The
RACECAR should automatically identify the color of a line it sees, then drive on the
center of the line throughout the obstacle course. The RACECAR should also identify
color changes, following colors with higher priority than others. Complete the lines 
of code under the #TODO indicators to complete the lab.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following keys:
- When the right trigger is pressed, the RACECAR moves forward at full speed
- When the left trigger is pressed, the RACECAR, moves backwards at full speed
- The angle of the RACECAR should only be controlled by the center of the line contour
- The RACECAR sees the color RED as the highest priority, then GREEN, then BLUE
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
MIN_CONTOUR_AREA = 30

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# TODO Part 1: Determine the HSV color threshold pairs for GREEN and RED
# Colors, stored as a pair (hsv_min, hsv_max) Hint: Lab E!
BLUE = ((90, 80, 50), (130, 255, 255))  # The HSV range for the color blue
GREEN = ((40, 36, 149), (84, 255, 255))  # The HSV range for the color green
RED = (((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (179, 255, 255)))  # The HSV range for the color red
ORANGE = ((1, 100, 149), (20, 255, 255))

# Color priority: Red >> Green >> Blue
#COLOR_PRIORITY = (RED, GREEN, BLUE)

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
prev_angle = 0
D = []
sum = 0

########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
def update_contour():
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0.0
        return

    # Crop the image to the floor directly in front of the car
    image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])


    contour = None
    orange_contours = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
    green_contours = rc_utils.find_contours(image, GREEN[0], GREEN[1])
    
    if orange_contours is not None:
        orange_contour = rc_utils.get_largest_contour(orange_contours, MIN_CONTOUR_AREA)
        contour = orange_contour
    if green_contours is not None:
        green_contour = rc_utils.get_largest_contour(green_contours, MIN_CONTOUR_AREA)
        if (contour is not None and green_contour is not None and rc_utils.get_contour_area(contour) > rc_utils.get_contour_area(green_contour)) or contour is None:
            contour = green_contour

    if contour is not None:
        contour_center = rc_utils.get_contour_center(contour)
        contour_area = rc_utils.get_contour_area(contour)
        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, contour_center)
    else:
        contour_center = None
        contour_area = 0.0
    #print(f"contour area: {contour_area}")
    rc.display.show_color_image(image)


def clamp(value: float, min: float, max: float) -> float:
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value
    
def remap_range(val: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    old_range = old_max - old_min
    new_range = new_max - new_min
    return new_range * (float(val-old_min) / float(old_range)) + new_min

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0.0
    angle = 0.0
    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)
    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)
    # Print start message
    print(
        ">> Lab 2A - Color Image Line Following\n"
        "\n"
        "Controls:\n"
        "   Right trigger = accelerate forward\n"
        "   Left trigger = accelerate backward\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():

    global speed
    global angle, prev_angle, D, sum
    update_contour()

    # TODO Part 3: Determine the angle that the RACECAR should receive based on the current 
    # position of the center of line contour on the screen. Hint: The RACECAR should drive in
    # a direction that moves the line back to the center of the screen.
    
    if contour_center is not None:
        #center line when visible
        SETPOINT = rc.camera.get_width() // 2
        error = SETPOINT - contour_center[1]
        if len(D) == 5:
            sum -= D.pop()
        D.append(error)
        sum += error
        #present_vue = c6ntour_center[1]
        kp = -0.0015#-0.002 #-0.0017#-0.0022 #-0.0015#-0.00275#-0.00325 #-0.0033 #-0.003125
        prev_angle = angle
        print(prev_angle)
        angle = kp * error + (-0.0001) * sum

        #if #abs(angle) <= 0.4:
            #angle = -0.001 * error #+ (-0.0008) * sum
        angle = rc_utils.clamp(angle, -1, 1)
    else:
        #turn at last known angle if the thing disappears
        #print(f"line gone: {prev_angle}")
        print('gone')
        if prev_angle < 0:
            angle = -1 #prev_angle - 0.4#1.0
        elif prev_angle > 0:
            angle = 1#prev_angle + 0.4
    speed = (0.97 - 0.55*(abs(angle)))
    angle = rc_utils.clamp(angle - 0.005, -1, 1)
    speed = rc_utils.clamp(speed, 0.6, 0.9)
    rc.drive.set_speed_angle(speed, angle)


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
