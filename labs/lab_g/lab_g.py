"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_g.py

Title: Lab G - Autonomous Parking

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: This script provides the RACECAR with the ability to autonomously detect an orange
cone and then drive and park 30cm away from the cone. Complete the lines of code under the 
#TODO indicators to complete the lab.

Expected Outcome: When the user runs the script, the RACECAR should be fully autonomous
and drive without the assistance of the user. The RACECAR drives according to the following
rules:
- The RACECAR detects the orange cone using its color camera, and can navigate to the cone
and park using its color camera and LIDAR sensors.
- The RACECAR should operate on a state machine with multiple states. There should not be
a terminal state. If there is no cone in the environment, the program should not crash.

Environment: Test your code using the level "Neo Labs > Lab G: Cone Parking".
Click on the screen to move the orange cone around the screen.
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import time

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
TARGET_AREA = 27448.0
MARGIN = 500.0
CROP_FLOOR = ((400, 0), (rc.camera.get_height(), rc.camera.get_width()))
# TODO Part 1: Determine the HSV color threshold pairs for ORANGE
ORANGE = ((10, 50, 50), (20, 255, 255))  # The HSV range for the color ORANGE


# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

# States
STATES = {1: "APPROACH", 2: "REVERSE", 3: "STOP", 4: "SEARCH"}
current_state = STATES[1]



########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
def search():
    global speed
    global angle
    while contour_area is None:
        rc.drive.set_speed_angle(1, 1)
        time.sleep(2)
        rc.drive.set_speed_angle(-1, -1)
        time.sleep(2)

def update_contour():
    global contour_center
    global contour_area
    global lrg_contour_o
    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        contours_o = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        lrg_contour_o = rc_utils.get_largest_contour(contours_o, MIN_CONTOUR_AREA)
        if lrg_contour_o is not None:
            area_o = rc_utils.get_contour_area(lrg_contour_o)
        else:
            pass
        

        if lrg_contour_o is not None:
            contour_center = rc_utils.get_contour_center(lrg_contour_o)
            contour_area = rc_utils.get_contour_area(lrg_contour_o)
        elif lrg_contour_o is None:
            contour_center = None
            contour_area = 0
            
    rc_utils.draw_circle(image, contour_center)
    rc.display.show_color_image(image)
    return contour_area != 0 and contour_center is not None
    
    # TODO Part 2: Complete this function by cropping the image to the bottom of the screen,
    # analyzing for contours of interest, and returning the center of the contour and the
    # area of the contour for the color of line we should follow (Hint: Lab 3)

# def search():
#     global speed
#     global angle
#     while contour_area is None:
#         rc.drive.set_speed_angle(1, 1)
#         time.sleep(2)
#         rc.drive.set_speed_angle(-1, -1)
#         time.sleep(2)

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(
        ">> Lab G - Autonomous Parking\n"
        "\n"
        "Controls:\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )

def bang_bang(line: int, setpoint: int) -> float:
    angle = 0
    error = setpoint - line
    if error < 0:
        angle = 1
    elif error > 0:
        angle = -1
    else:
        angle = 0
    return angle

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    update_contour()

    if contour_area == 0:
        current_state = "SEARCH"
    elif contour_area < TARGET_AREA - MARGIN: # too far
        current_state = "APPROACH"
    elif contour_area > TARGET_AREA + MARGIN: # too close
        current_state = "REVERSE"
    else:
        current_state = "STOP"

    if contour_center is not None:
        setpoint = rc.camera.get_width() // 2
        present_value = contour_center[1]
        kp = 0.003125
        error = present_value - setpoint # setpoint - present_value
        angle = kp * error
        angle = rc_utils.clamp(angle, -1, 1)

    if current_state == "APPROACH":
        speed = 0.5
        # approach cone
    elif current_state == "REVERSE":
        speed = -0.5
        # reverse from cone
    elif current_state == "STOP":
        speed = 0
        # stop the car
    elif current_state == "SEARCH":
        speed = 0.35
        angle = 1
        # search for cone
        # speed = 1, angle = 1, then speed = -1, angle = -1
    if contour_center is not None:
        setpoint = rc.camera.get_width() // 2
        present_value = contour_center[1]
        kp = 0.003125
        error = present_value - setpoint # setpoint - present_value
        angle = kp * error
        angle = rc_utils.clamp(angle, -1, 1)

    # if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0:
    #     speed = 1
    # elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0:
    #     speed = -1
    # if contour_area == 0:
    #     search()
    # elif contour_area < TARGET_AREA - MARGIN: # too far
    #     speed = 0.5
    # elif contour_area > TARGET_AREA + MARGIN: # too close
    #     speed = -0.5
    # else:
    #     speed = 0

    # if speed < -1 :
    #     speed = -1
    # elif speed > 1:
    #     speed = 1
    # else:
    #     pass
    # Search for contours in the current color image

    # TODO Part 3: Park the car 30cm away from the closest orange cone.
    # You may use a state machine and a combination of sensors (color camera,
    # or LIDAR to do so). Depth camera is not allowed at this time to match the
    # physical RACECAR Neo.

    # Set the speed and angle of the RACECAR after calculations have been complete
    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)


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
