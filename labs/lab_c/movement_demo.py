"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: movement_demo.py

Title: RACECAR Movement Demo

Author: Kripa Sudhir

Purpose: To demonstrate movement commands for the RACECAR

Expected Outcome: Move the RACECAR using the following controller inputs:
- When the "A" button is pressed, the RACECAR moves forward at full speed
- When the "B" button is pressed, the RACECAR moves backward at full speed
- When the "X" button is pressed, the RACECAR moves forward and to the right
at full speed
- When the "Y" button is pressed, the RACECAR moves forward and to the left
at full speed
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
speed = 0
angle = 0

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    # Initial speed and angle are both 0. The car should not move.
    rc.drive.set_speed_angle(speed, angle)
# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    # When A is pressed, drive forward full speed
    if rc.controller.is_down(rc.controller.Button.A):
        speed = 1
        angle = 0
    elif rc.controller.is_down(rc.controller.Button.B):
        speed = -1
    elif rc.controller.is_down(rc.controller.Button.X):
        speed = 1
        angle = -1
    elif rc.controller.is_down(rc.controller.Button.Y):
        speed = 1
        angle = 1
    else:
        speed = 0
        angle = 0
    print(f"Speed = {speed}, Angle = {angle}")
    rc.drive.set_speed_angle(speed, angle)
    

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
