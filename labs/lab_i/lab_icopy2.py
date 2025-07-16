"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_i.py

Title: Lab I - Wall Follower

Author: [stuti] << [Write your name or team name here]

Purpose: This script provides the RACECAR with the ability to autonomously follow a wall.
The script should handle wall following for the right wall, the left wall, both walls, and
be flexible enough to handle very narrow and very wide walls as well.

Expected Outcome: When the user runs the script, the RACECAR should be fully autonomous
and drive without the assistance of the user. The RACECAR drives according to the following
rules:
- The RACECAR detects a wall using the LIDAR sensor a certain distance and angle away.
- Ideally, the RACECAR should be a set distance away from a wall, or if two walls are detected,
should be in the center of the walls.
- The RACECAR may have different states depending on if it sees only a right wall, only a 
left wall, or both walls.
- Both speed and angle parameters are variable and recalculated every frame. The speed and angle
values are sent once at the end of the update() function.

Note: This file consists of bare-bones skeleton code, which is the bare minimum to run a 
Python file in the RACECAR sim. Less help will be provided from here on out, since you've made
it this far. Good luck, and remember to contact an instructor if you have any questions!

Environment: Test your code using the level "Neo Labs > Lab I: Wall Follower".
Use the "TAB" key to advance from checkpoint to checkpoint to practice each section before
running through the race in "race mode" to do the full course. Lowest time wins!
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
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

walls_detected = ""
queue = [] # The queue of instructions
state = ""
stay_dist = 50

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0.5
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle, state
    global check_right, error
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
        speed = 0.5

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
    if abs(error) > 20:
        kp_a = 0.8
    else: 
        kp_a = 0.005
    angle = kp_a * error 
    angle = max(min(angle,1),-1)

    kp_s = 0.5
    raw_speed = kp_s * forward_distance
    if forward_distance > 40:
        speed = max(min(raw_speed,1),0.1)
    else:
        speed = 0.1
    
    if abs(angle) > 0.7:
        speed *= 0.95

    rc.drive.set_speed_angle(speed, angle)



# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    print("check_right = " + str(check_right))
    print(state)
    print(speed)
    print("hit an error of" + str(error))

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
