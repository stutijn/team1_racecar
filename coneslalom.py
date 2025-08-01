"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: coneslalom.py

Title: Cone Slalom

Author: Team 1

Purpose: Detect cones of different colors and manuever around each one accordingly. 
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../../library')
import racecar_core
import racecar_utils as rc_utils
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
speed = 0.5
angle = 0
RED = (((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (179, 255, 255)))
BLUE = ((90, 150, 30), (130, 255, 255))
#BLUE = ((90, 100, 50), (130, 255, 255))
MIN_CONTOUR_AREA = 1000
contour_area = 0
contour_center = 0
color = ""
prev_color = None
kp = 0.003
changing = 0
turn_direction = 0.81
state = "tracking"
turn_direction = 1
timer = 0
FORWARD = 14
CENTER_TOLERANCE = 40  # How centered the cone should be
AREA_THRESHOLD = 2000  # How big the cone should be to start turning
########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    
    pass # Remove 'pass' and write your source code for the start() function here

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed, angle, contour_center, contour_area, color, prev_color
    global state, turn_direction, timer

    update_contour()
    acceleration = rc.physics.get_linear_acceleration()
    forward_acceleration = acceleration[1]
    if abs(forward_acceleration) < 9.79:
        rc.drive.set_max_speed(0.25)
    else:
        rc.drive.set_max_speed(0.28)
    print(state, contour_area, prev_color, color)

    if state == "tracking":
        print("tracking")
        if contour_area < 2500:
            rc.drive.set_speed_angle(0.7, 0)
            print("GOING FORWARD")
        if contour_center is not None:
            cx = contour_center[1]
            image_center = rc.camera.get_width() // 2
            error = cx - image_center

            angle = error * 0.05
            angle = rc_utils.clamp(angle, -1, 1)

            rc.drive.set_speed_angle(0.55, angle)

            if abs(error) < CENTER_TOLERANCE and contour_area > AREA_THRESHOLD:
                if color == "RED":
                    turn_direction = 0.81 
                else:
                    turn_direction = -0.81  
                state = "turning"

        else:
            state = "tracking"

    elif state == "turning":
        print("turning")
        if(contour_area > 4500):
            if turn_direction < 0:
                rc.drive.set_speed_angle(0.75, -1)
            else:
                rc.drive.set_speed_angle(0.75, 1)
        else:
            rc.drive.set_speed_angle(0.75, turn_direction)

        if contour_center is None:
            state = "forward"
            timer = FORWARD
            rc.drive.set_speed_angle(0.55, 0)
            
    elif state == "forward":
        print("forward")
        if contour_center is not None:
                state = "tracking"
        if timer > 0:
            if contour_center is not None and ((color != prev_color and contour_area > AREA_THRESHOLD)):

                if color == "RED":
                    turn_direction = 0.78
                else:
                    turn_direction = -0.78
                state = "turning"
                rc.drive.set_speed_angle(0.6, -turn_direction)
                timer = 0
            else:
                rc.drive.set_speed_angle(0.65, 0)
            timer -= 1
        else:
            rc.drive.set_speed_angle(0.62, -turn_direction)

        prev_color = color

def update_contour():
    global color, contour_area, contour_center

    image = rc.camera.get_color_image()
    if image is None:
        contour_center = None
        contour_area = 0
        return

    contour = None

    blue_contours = rc_utils.find_contours(image, BLUE[0], BLUE[1])
    red_contours_1 = rc_utils.find_contours(image, RED[0][0], RED[0][1])
    red_contours_2 = rc_utils.find_contours(image, RED[1][0], RED[1][1])
    red_contours = red_contours_1 + red_contours_2
    red_max = rc_utils.get_largest_contour(red_contours, MIN_CONTOUR_AREA)
    blue_max = rc_utils.get_largest_contour(blue_contours, MIN_CONTOUR_AREA)
    if red_max is None:
        contour = blue_max
        color = "BLUE"
    elif blue_max is None:
        contour = red_max
        color = "RED"
    elif rc_utils.get_contour_area(red_max) > rc_utils.get_contour_area(blue_max):
        contour = red_max
        color = "RED"
    else:
        contour = blue_max
        color = "BLUE"
    
    contour_center = rc_utils.get_contour_center(contour)
    if contour is not None:
        contour_area = rc_utils.get_contour_area(contour)

        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, contour_center)

    # Always show image

    rc.display.show_color_image(image)

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


