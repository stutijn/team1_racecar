"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: grand_prix.py

Title: Grand Prix Day!

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: Write a script to enable fully autonomous behavior from the RACECAR. The
RACECAR will traverse the obstacle course autonomously without human intervention.
Once the start button is pressed, the RACECAR must drive through the course until it
reaches finish line.

Note: There is no template code in this document to follow except for the RACECAR script 
structure found in template.py. You are expected to use code written from previous labs
to complete this challenge. Good luck!

Expected Outcome: When the user runs the script, they must not be able to manually control
the RACECAR. The RACECAR must move forward on its own, traverse through the course, and then
stop on its own.
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

rc = racecar_core.create_racecar()

# Declare any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

walls_detected = ""
queue = [] # The queue of instructions
state = ""
stay_dist = 50

########################################################################################
# Functions
########################################################################################

def wall_follower():
    global speed
    global angle, state
    global check_right
    stay_dist = 50
    # Access the most recent lidar scan.
    scan = rc.lidar.get_samples()

    # Get the distance of the measurement directly in front of the car
    _, forward_distance = rc_utils.get_lidar_closest_point(scan, (335, 15))
    check_right = rc_utils.get_lidar_average_distance(scan, 90, 90)
    check_left = rc_utils.get_lidar_average_distance(scan, 270, 90)

    max_wall_dist = 170

    right_wall = check_right < max_wall_dist
    left_wall = check_left < max_wall_dist

    if right_wall and left_wall:
        state = "center"
    elif right_wall:
        state = "right"
    elif left_wall:
        state = "left"
    else:
        state = "no_walls"
        error = 0
        speed = 0.3

    if state == "center":
        error = check_right - check_left

    elif state == "right":
        if check_right < 45: #if really close, keep stay distance smaller for tight turns
            stay_dist = 50
        else:
            stay_dist = 25
        error = stay_dist - check_right

    elif state == "left":
        if check_left < 45:
            stay_dist = 50
        else:
            stay_dist = 25
        error = stay_dist - check_left

    elif state == "no_walls":
        error = stay_dist

    #control
    if abs(error) > 30:
        kp_a = 0.15
    else: 
        kp_a = 0.08
    angle = kp_a * error 
    angle = max(min(angle,1),-1)

    kp_s = 0.008
    raw_speed = kp_s * forward_distance
    if forward_distance > 40:
        speed = max(min(raw_speed,1),0.1)
    else:
        speed = 0.1
    
    if abs(angle) > 0.7:
        speed *= 0.85

    rc.drive.set_speed_angle(speed, angle)

def color_follower():
    pass


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle
    # Initialize variables
    speed = 0.5
    angle = 0

    #TODO: set first AR marker detected as the corresponding one to wall follower

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle) 

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():

    #TODO: detect ARUco markers; depending on whats detect, identify state. based on diff state, stay in that area UNTIL new marker detect
    # basically save the old marker as current marker until new marker detected.
    wall_follower () # Remove 'pass' and write your source code for the update() function here

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
