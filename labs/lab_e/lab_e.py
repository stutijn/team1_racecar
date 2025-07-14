"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_e.py

Title: Lab E - Stoplight Challenge

Author: Kripa Sudhir << [Write your name or team name here]

Purpose: Write a script to enable autonomous behavior from the RACECAR. When
the RACECAR sees a stoplight object (colored cube in the simulator), respond accordingly
by going straight, turning right, turning left, or stopping. Append instructions to the
queue depending on whether the position of the RACECAR relative to the stoplight reaches
a certain threshold, and be able to respond to traffic lights at consecutive intersections. 

Expected Outcome: When the user runs the script, the RACECAR should control itself using
the following constraints:
- When the RACECAR sees a BLUE traffic light, make a right turn at the intersection
- When the RACECAR sees an ORANGE traffic light, make a left turn at the intersection
- When the RACECAR sees a GREEN traffic light, go straight
- When the RACECAR sees a RED traffic light, stop moving,
- When the RACECAR sees any other traffic light colors, stop moving.

Considerations: Since the user is not controlling the RACECAR, be sure to consider the
following scenarios:
- What should the RACECAR do if it sees two traffic lights, one at the current intersection
and the other at the intersection behind it?
- What should be the constraint for adding the instructions to the queue? Traffic light position,
traffic light area, or both?
- How often should the instruction-adding function calls be? Once, twice, or 60 times a second?

Environment: Test your code using the level "Neo Labs > Lab 3: Stoplight Challenge".
By default, the traffic lights should direct you in a counterclockwise circle around the course.
For testing purposes, you may change the color of the traffic light by first left-clicking to 
select and then right clicking on the light to scroll through available colors.
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
# The smallest contour we will recognize as a valid contour (Adjust threshold!)
MIN_CONTOUR_AREA = 30

# CROP_FLOOR = ((150, 290), (rc.camera.get_height(), 350))
# TODO Part 1: Determine the HSV color threshold pairs for ORANGE, GREEN, RED, YELLOW, and PURPLE
# Colors, stored as a pair (hsv_min, hsv_max)
BLUE = ((90, 250, 250), (120, 255, 255))  # The HSV range for the color blue
GREEN = ((35, 50, 50), (75, 255, 255))  # The HSV range for the color green
RED = ((170, 50, 50), (5, 255, 255))
ORANGE = ((10, 50, 50), (20, 255, 255)) # The HSV range for the color orange
YELLOW = ((21, 50, 50), (29, 255, 255)) # The HSV range for the color yellow
PURPLE = ((130, 50, 50), (160, 255, 255)) # The HSV range for the color purple

# >> Variables
speed = 0
angle = 0
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

queue = [] # The queue of instructions
stoplight_color = "" # The current color of the stoplight
last_color = "" # Last seen color

########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
def update_contour():
    global contour_center
    global contour_area
    global stoplight_color
    global areas
    areas = []
    area_b = area_r = area_g = area_o = area_y = area_p = 0
    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
        # TODO Part 2: Search for line colors, and update the global variables
        # contour_center and contour_area with the largest contour found
        contours_b = rc_utils.find_contours(image, BLUE[0], BLUE[1])
        contours_o = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contours_g = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        contours_r = rc_utils.find_contours(image, RED[0], RED[1])
        contours_y = rc_utils.find_contours(image, YELLOW[0], YELLOW[1])
        contours_p = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])
        if contours_b is not None:
            lrg_contour_b = rc_utils.get_largest_contour(contours_b, MIN_CONTOUR_AREA)
            if lrg_contour_b is not None:
                area_b = rc_utils.get_contour_area(lrg_contour_b)
                areas.append(area_b)
        else:
            pass
        if contours_r is not None:
            lrg_contour_r = rc_utils.get_largest_contour(contours_r, MIN_CONTOUR_AREA)
            if lrg_contour_r is not None:
                area_r = rc_utils.get_contour_area(lrg_contour_r)
                areas.append(area_r)
        else:
            pass
        if contours_g is not None:
            lrg_contour_g = rc_utils.get_largest_contour(contours_g, MIN_CONTOUR_AREA)
            if lrg_contour_g is not None:
                area_g = rc_utils.get_contour_area(lrg_contour_g)
                areas.append(area_g)
        else:
            pass
        if contours_o is not None:
            lrg_contour_o = rc_utils.get_largest_contour(contours_o, MIN_CONTOUR_AREA)
            if lrg_contour_o is not None:
                area_o = rc_utils.get_contour_area(lrg_contour_o)
                areas.append(area_o)
        else:
            pass
        if contours_y is not None:
            lrg_contour_y = rc_utils.get_largest_contour(contours_y, MIN_CONTOUR_AREA)
            if lrg_contour_y is not None:
                area_y = rc_utils.get_contour_area(lrg_contour_y)
                areas.append(area_y)
        else:
            pass
        if contours_p is not None:
            lrg_contour_p = rc_utils.get_largest_contour(contours_p, MIN_CONTOUR_AREA)
            if lrg_contour_p is not None:
                area_p = rc_utils.get_contour_area(lrg_contour_p)
                areas.append(area_p)
        else:
            pass
        
        # TODO Part 3: Repeat the search for all potential traffic light colors,
        # then select the correct color of traffic light detected.
        if len(areas) == 0:
            stoplight_color = 'none'
            return
        else:
            if max(areas) == area_b:
                stoplight_color = 'blue'
                contour_center = rc_utils.get_contour_center(lrg_contour_b)
                contour_area = rc_utils.get_contour_area(lrg_contour_b)
                rc_utils.draw_contour(image, lrg_contour_b)
            elif max(areas) == area_r:
                stoplight_color = 'red'
                contour_center = rc_utils.get_contour_center(lrg_contour_r)
                contour_area = rc_utils.get_contour_area(lrg_contour_r)
                rc_utils.draw_contour(image, lrg_contour_r)
            elif max(areas) == area_g:
                stoplight_color = 'green'
                contour_center = rc_utils.get_contour_center(lrg_contour_g)
                contour_area = rc_utils.get_contour_area(lrg_contour_g)
                rc_utils.draw_contour(image, lrg_contour_g)
            elif max(areas) == area_o:
                stoplight_color = 'orange'
                contour_center = rc_utils.get_contour_center(lrg_contour_o)
                contour_area = rc_utils.get_contour_area(lrg_contour_o)
                rc_utils.draw_contour(image, lrg_contour_o)
            elif max(areas) == area_y or max(areas) == area_p:
                stoplight_color = 'other'
                if max(areas) == area_y:
                    contour_center = rc_utils.get_contour_center(lrg_contour_y)
                    rc_utils.draw_contour(image, lrg_contour_y)
                else:
                    contour_area = rc_utils.get_contour_area(lrg_contour_p)
                    rc_utils.draw_contour(image, lrg_contour_p)
        print(f"Red area: {area_r}; Blue area; {area_b}")
        
        
        # Display the image to the screen
        rc.display.show_color_image(image)

# [FUNCTION] The start function is run once every time the start button is pressed
def start():

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(0,0)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message (You may edit this to be more informative!)
    print(
        ">> Lab 3 - Stoplight Challenge\n"
        "\n"
        "Controls:\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global queue
    global speed
    global angle
    global last_color
    update_contour()

    # TODO Part 2: Complete the conditional tree with the given constraints.
    if stoplight_color != last_color and len(queue) == 0:
        if stoplight_color == 'blue':
            turnRight()
            last_color = 'blue'
            areas.clear()
            # Call the correct function to append the instructions to the list
        elif stoplight_color == 'orange':
            turnLeft()
            last_color = 'orange'
            areas.clear()
        elif stoplight_color == 'green':
            goStraight()
            last_color = 'green'
            areas.clear()
        elif stoplight_color == 'red' or stoplight_color == 'other':
            stopNow()
            areas.clear()
            if stoplight_color == 'red':
                last_color = 'red'
            else:
                last_color = 'other'
    print(f"Detected stoplight color: {stoplight_color}; Queue length: {len(queue)}")
        # Call the correct function to append the instructions to the list
    if len(queue) > 0:
        speed = queue[0][1]
        if queue[0][2] == 0 and contour_center is not None:
            setpoint = rc.camera.get_width() // 2
            present_value = contour_center[1]
            kp = 0.003125
            error = present_value - setpoint # setpoint - present_value
            angle = kp * error
            angle = rc_utils.clamp(angle, -1, 1)
        else:
            angle = queue[0][2]
        queue[0][0] -= rc.get_delta_time()
        if queue[0][0] <= 0:
            queue.pop(0)
    
    # ... You may need more elif/else statements
    if last_color == stoplight_color and len(queue) == 0:
        print("Reset time!")
        last_color = 'none'
    # TODO Part 3: Implement a way to execute instructions from the queue once they have been placed
    # by the traffic light detector logic (Hint: Lab 2)

    # Send speed and angle commands to the RACECAR
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
    print(f"Current: {stoplight_color}; Last: {last_color}; Queue: {len(queue)}")
# [FUNCTION] Appends the correct instructions to make a 90 degree right turn to the queue
def turnRight():
    global queue

    # TODO Part 4: Complete the rest of this function with the instructions to make a right turn
    queue.clear()
    queue.append([2.05, 1, 0])
    queue.append([1.34, 1, 1])
    # queue.append([1.5, 1, 0])
    queue.append([0.5, 0, 0])
# [FUNCTION] Appends the correct instructions to make a 90 degree left turn to the queue
def turnLeft():
    global queue
    
    # TODO Part 5: Complete the rest of this function with the instructions to make a left turn
    queue.clear()
    queue.append([2.05, 1, 0])
    queue.append([1.29, 1, -1])
    # queue.append([1.5, 1, 0])
    queue.append([0.5, 0, 0])
# [FUNCTION] Appends the correct instructions to go straight through the intersectionto the queue
def goStraight():
    global queue

    # TODO Part 6: Complete the rest of this function with the instructions to make a left turn
    queue.clear()
    queue.append([3.3, 1, 0])
    queue.append([0.5, 0, 0])
# [FUNCTION] Clears the queue to stop all actions
def stopNow():
    global queue
    queue.clear()
    # queue.append([0.5, 1, 0])
    queue.append([0.5, 0, 0])
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