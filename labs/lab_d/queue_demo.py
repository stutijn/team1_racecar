"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: queue_demo.py

Title: Queue Demo

Author: Kripa Sudhir

Purpose: Demonstrate queues in semi-autonomous driving scheme for
RACECAR by adding instructionss and running them in the update() function

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
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
queue = []
speed = 0
angle = 0

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
    global queue
    global speed
    global angle

    # when A is pressed, add instructions for circle
    if rc.controller.was_pressed(rc.controller.Button.A):
        drive_circle()

    # If queue isn't empty, follow current instruction
    if len(queue) > 0:
        speed = queue[0][1]
        angle = queue[0][2]
        queue[0][0] -= rc.get_delta_time()
        if queue[0][0] <= 0:
            queue.pop(0)
            queue.append([0.5, 0, 0])
    
    
    rc.drive.set_speed_angle(speed, angle)

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here

# When called, clear the queue, then place instruction
# inside the queue, causing the RACECAR to drive in circle
def drive_circle():
    global queue

    CIRCLE_TIME = 5.5
    BRAKE_TIME = 0.5
    
    queue.clear()

    # Add instructions to queue
    queue.append([5.5, 1, 1])
    queue.append([0.5, -1, 1])

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
