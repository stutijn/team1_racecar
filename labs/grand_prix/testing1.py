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
global speed

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    # global speed

    speed = 0.0 # The initial speed is at 1.0

    # This tells the car to begin at a standstill
    rc.drive.stop()

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    if rc.controller.was_pressed(rc.controller.Button.A):
        increase_speed()
    elif rc.controller.was_pressed(rc.controller.Button.B):
        decrease_speed()
    else:
        speed = 0

    # Send the speed and angle values to the RACECAR
    print(f"Speed: {speed}")
    rc.drive.set_speed_angle(speed, 0)

def increase_speed():
    global speed
    speed += 0.1

def decrease_speed():
    global speed
    speed -= 0.1

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()