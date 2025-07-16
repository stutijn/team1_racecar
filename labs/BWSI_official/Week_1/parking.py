"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_g.py

Title: Lab G - Autonomous Parking

Author: [stuti] << [Write your name or team name here]

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

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(2, "../../../library")
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
CROP_FLOOR = ((rc.camera.get_height()//2, 0), (rc.camera.get_height(), rc.camera.get_width()))

# TODO Part 1: Determine the HSV color threshold pairs for ORANGE
# Colors, stored as a pair (hsv_min, hsv_max) 
ORANGE = ((1,75,100),(60,255,255))

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

state = "teleop"
PARKING_DISTANCE = 275  # in cm
SLOW_ZONE = 500               # Start slowing down here
DIST_TOLERANCE = 0.75           # Acceptable range (e.g. 27-33 cm)
BACKING_THRESHOLD = 48.9        # If we get closer than this, back up

########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
# def update_contour():
#     global contour_center
#     global contour_area

#     image = rc.camera.get_color_image()
#     image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

#     # TODO Part 2: Complete this function by cropping the image to the bottom of the screen,
#     # analyzing for contours of interest, and returning the center of the contour and the
#     # area of the contour for the color of line we should follow (Hint: Lab 3)

#     if image is None:
#         contour_center = None
#         contour_area = 0
#     else:
#         hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                                           
#         contours = rc_utils.find_contours(hsv,ORANGE[0],ORANGE[1]) #searches for all contours of current color

#         contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA) #select largest contour
#         if contour is not None:
#             contour_center = rc_utils.get_contour_center(contour)
#             contour_area= rc_utils.get_contour_area(contour)
#             rc_utils.draw_contour(image,contour)
          
#         # Display the image to the screen
#             rc.display.show_color_image(image)

#     rc.display.show_color_image(image)

def update_lidar():
    global distance 

    scan = rc.lidar.get_samples()
    distance = rc_utils.get_lidar_average_distance(scan, 0, 20)


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
    rc.drive.set_max_speed(1)
    rc.set_update_slow_time(0.5)


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    global state, distance
    #update_contour()
    update_lidar()
    
    #print(f'area: {contour_area} and the distance: {distance}')
    print(f"distance: {distance}")
    #SETING MODE
    if (distance - PARKING_DISTANCE) <= DIST_TOLERANCE:
        state = "parking"

    # elif distance < BACKING_THRESHOLD: #if closer than 25 cm, back away
    #     state = "backing"

    elif distance <= SLOW_ZONE: #if under 150 cm away
        state = "approaching_slow"
    
    else:
        state = "teleop"
    #MODES DO STUFF
    if state == "teleop": 
        if distance < 250: #and contour_area > 40000:
            state = "approaching_slow"
        else:
            (y, x) = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
            (b, a) = rc.controller.get_joystick(rc.controller.Joystick.RIGHT)
            speed =  x 
            angle = b
            angle += 0.1
            angle = rc_utils.clamp(angle,-1,1)
   
    if state == "approaching_slow":     
        print(distance)
        speed = -1
        angle = 0

    # elif state == "backing":
    #     speed = -0.25
    #     angle = rc_utils.clamp((contour_center[1] - rc.camera.get_width() // 2) / 100, -1, 1)

    elif state == "parking":
        speed = 0
        angle = 0
    print(state)

    rc.drive.set_speed_angle(speed, angle)

    '''
    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)'''


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
