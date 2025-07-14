"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_f.py

Title: Lab F - Line Follower

Author: [PLACEHOLDER] << [Write your name or team name here]

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
MIN_CONTOUR_AREA = 500

# A crop window for the floor directly in front of the car
# CROP_FLOOR = ((2 * rc.camera.get_height()//3 - 50, 2 * rc.camera.get_height() //3), (rc.camera.get_height(), rc.camera.get_width()))
CROP_FLOOR = ((2 * rc.camera.get_height()//3 - 50, 0), (rc.camera.get_height(), rc.camera.get_height() //3))

# TODO Part 1: Determine the HSV color threshold pairs for GREEN and RED
# Colors, stored as a pair (hsv_min, hsv_max) Hint: Lab E!
BLUE = ((90, 50, 50), (120, 255, 255))  # The HSV range for the color blue
GREEN = ((35, 50, 50), (75, 255, 255))  # The HSV range for the color green
RED = ((170, 50, 50), (5, 255, 255))  # The HSV range for the color red
ORANGE = ((10, 50, 50), (20, 255, 255)) # HSV range for orange
PURPLE = ((130, 50, 50), (160, 255, 255)) # HSV range for purple

# Color priority: Red >> Green >> Blue
COLOR_PRIORITY = (RED, GREEN, BLUE)

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row (y), pixel column (x)) of contour
contour_area = 0  # The area of contour


########################################################################################
# Functions
########################################################################################

def remap_range(val: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    old_range = old_max - old_min
    new_range = new_max - new_min
    return new_range * (float(val - old_min) / float(old_range)) + new_min

def clamp(value: float, min: float, max: float) -> float:
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value
    
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

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
def update_contour():
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()
    image_1 = rc.camera.get_color_image()
    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        # TODO Part 2: Search for line colors, and update the global variables
        # contour_center and contour_area with the largest contour found

        # Search for contours of the current color
        contours_o = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contours_p = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])
        # contours_g = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        lrg_contour_o = rc_utils.get_largest_contour(contours_o, MIN_CONTOUR_AREA)
        if lrg_contour_o is not None:
            area_o = rc_utils.get_contour_area(lrg_contour_o)
        else:
            area_o = 0
        lrg_contour_p = rc_utils.get_largest_contour(contours_p, MIN_CONTOUR_AREA)
        if lrg_contour_p is not None:
            area_p = rc_utils.get_contour_area(lrg_contour_p)
        else:
            area_p = 0
        # lrg_contour_g = rc_utils.get_largest_contour(contours_g, MIN_CONTOUR_AREA)

        if lrg_contour_o is not None and area_o > area_p:
            contour_center = rc_utils.get_contour_center(lrg_contour_o)
            contour_area = rc_utils.get_contour_area(lrg_contour_o)
            rc_utils.draw_circle(image, contour_center)
        if lrg_contour_p is not None and area_p > area_o:
            contour_center = rc_utils.get_contour_center(lrg_contour_p)
            contour_area = rc_utils.get_contour_area(lrg_contour_p)
            rc_utils.draw_circle(image, contour_center)
        elif lrg_contour_p is None and lrg_contour_o is None:
            contour_center = None
            contour_area = 0
        # elif lrg_contour_b is not None:
        #     contour_center = rc_utils.get_contour_center(lrg_contour_b)
        #     contour_area = rc_utils.get_contour_area(lrg_contour_b)
        
        # Display the image to the screen
        rc.display.show_color_image(image)

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle
    global speed_offset
    speed_offset = 0.1 # The initial speed offset is 0.5
    
    # Initialize variables
    speed = 0
    angle = 0

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
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    update_contour()
    if contour_center is not None and contour_center[1] > 50 and contour_center[0] < 100:
        setpoint = 90
        present_value = contour_center[1]
        kp = -0.003125
        error = present_value - setpoint # setpoint - present_value
        angle = kp * error
        angle = rc_utils.clamp(angle, -1, 1)
    elif contour_center is None:
        angle = -1
        print("No contour detected")
    elif contour_center[0] > 100:
        angle = 1
        print(contour_center[0])
        
        
    # print(contour_area)
    # if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0:
    #     speed = .5
    # elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0:
    #     speed = -.5
    speed = 0.5
    # if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0 or rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0:
    #     if rc.controller.was_pressed(rc.controller.Button.A):
    #         speed += speed_offset
    #     elif rc.controller.was_pressed(rc.controller.Button.B):
    #         speed -= speed_offset
    #     else:
    #         pass
    
    
    # if speed < -1 :
    #     speed = -1
    # elif speed > 1:
    #     speed = 1
    # else:
    #     pass

    rc.drive.set_speed_angle(speed, angle)
    # Search for contours in the current color image
    

    # TODO Part 3: Determine the angle that the RACECAR should receive based on the current 
    # position of the center of line contour on the screen. Hint: The RACECAR should drive in
    # a direction that moves the line back to the center of the screen.

    # Choose an angle based on contour_center
    # If we could not find a contour, keep the previous angle
    if contour_center is not None:
        # angle = _____
        pass

    # Use the triggers to control the car's speed
    # rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    # lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    # speed = rt - lt

    # rc.drive.set_speed_angle(speed, angle)

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
    # if rc.camera.get_color_image() is None:
    #     # If no image is found, print all X's and don't display an image
    #     print("X" * 10 + " (No image) " + "X" * 10)
    # else:
    #     # If an image is found but no contour is found, print all dashes
    #     if contour_center is None:
    #         print("-" * 32 + " : area = " + str(contour_area))

    #     # Otherwise, print a line of dashes with a | indicating the contour x-position
    #     else:
    #         s = ["-"] * 32
    #         s[int(contour_center[1] / 20)] = "|"
    #         print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
