"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: controller_testing.py << [Modify with your own file name!]

Title: Contorller Testing << [Modify with your own title]

Author: Stuti J << [Write your name or team name here]

Purpose: Verify functions from the controller library and observe output of such functions << [Write the purpose of the script here]

Expected Outcome: Test button, tirgger, and joystick commands to verify their output. 
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
    if rc.controller.is_down(rc.controller.Button.A):
        l_trigger = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        r_trigger = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        print(f"Left Trigger = {l_trigger} || Right Trigger = {r_trigger}")

    if rc.controller.is_down(rc.controller.Button.B):
        (x, y)= rc.controller.get_joystick(rc.controller.Joystick.RIGHT)
        print(f"Right Joystick: x = {x}, y = {y}")

    if rc.controller.was_pressed(rc.controller.Button.X):
        print(f"The X button is pressed!")

    if rc.controller.was_released(rc.controller.Button.Y):
        print(f"The Y button is pressed!")

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
