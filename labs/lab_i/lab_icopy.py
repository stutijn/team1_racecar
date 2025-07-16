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
stay_dist = 30
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
    global walls_detected
    global check_right
    angle = 0
    # Access the most recent lidar scan.
    scan = rc.lidar.get_samples()

    # Get the distance of the measurement directly in front of the car
    forward_distance = scan[0]
    check_right = rc_utils.get_lidar_average_distance(scan,90,90)

    
    #if check_right is not None:
    check_left = rc_utils.get_lidar_average_distance(scan,270,90)
    if check_left is not None:
        walls_detected = "TWO"
            
       # else:
        #    walls_detected = "RIGHT"
  #  else:
      #  walls_detected = "LEFT"

    if walls_detected == "TWO":
        state = "balance"
    elif walls_detected == "RIGHT":
        state = "right"
    elif walls_detected == "LEFT":
        state = "left"
    else:
        print("State failure.") 

    if state == "balance":
        #ensure between the two detected scans
        stay_dist = (check_right + check_left)/2
        if check_right > check_left: #the distance from the right side is greater than the distance from the left
            if check_right > (check_left * 2):
                angle +=.25 #move closer to the right
            else:
                angle +=.55
       
        else:
            if check_left > (check_right * 2):
                angle -=.25
            else:
                angle -=.45
    

    elif state == "right":
        #ensure
        if check_right > stay_dist:
            angle += .1
        elif check_right < stay_dist:
            angle -= .1

    elif state == "left":
        if check_right > stay_dist:
            angle -= .1
        elif check_right < stay_dist:
            angle += .1
       
    else:
        print("fix ur code")

    rc.drive.set_speed_angle(speed, angle)



# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    print(check_right) 
    print("detected" + walls_detected) # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
