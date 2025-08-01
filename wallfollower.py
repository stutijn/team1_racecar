"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: wallfollower.py

Title: Wall Follower

Author: TEAM 1

Purpose: Find a setpoint between two walls and drive between them by maintaining setpoint using PID control. 
"""

########################################################################################
# Imports
########################################################################################

import sys

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
speed = 0
angle = 0
D = []
sum = 0

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
    global speed, angle
    global D, sum
    scan = rc.lidar.get_samples()
    left = rc_utils.get_lidar_average_distance(scan, 320, 30)
    right = rc_utils.get_lidar_average_distance(scan, 40, 30)
    front = rc_utils.get_lidar_average_distance(scan, 0, 30)
    Kp = -0.0031 
    Kd = -0.0002
    error = left - right
    if len(D) == 5:
        sum -= D.pop() 
    D.append(error)
    sum += error
    angle = Kp * error + Kd * (error / 5) 
    if front < 210:
        left = rc_utils.get_lidar_average_distance(scan, 60, 30)
        right = rc_utils.get_lidar_average_distance(scan, 300, 30)
        if right - left > 45 and (left > 200 or right > 200): 
            angle = -1
        elif left - right > 45 and (left > 200 or right > 200):
            angle = 1
    angle = rc_utils.clamp(angle, -1, 1)
    speed = 1/(1+0.55*(abs(angle)))
    speed = rc_utils.clamp(speed, 0.7, 1)
  
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
