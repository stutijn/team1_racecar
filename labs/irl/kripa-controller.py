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
global speed
global angle
global speed_offset
global angle_offset

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle
    global speed_offset
    global angle_offset

    speed = 0.0 # The initial speed is at 1.0
    angle = 0.0 # The initial turning angle away from the center is at 0.0
    speed_offset = 0.1 # The initial speed offset is 0.5
    angle_offset = 0.1 # The inital angle offset is 1.0

    # This tells the car to begin at a standstill
    rc.drive.stop()

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    global speed_offset
    global angle_offset
    
    # TODO Part 1: Modify the following conditional statement such that when the
    # right trigger is pressed, the RACECAR moves forward at the designated speed.
    # when the left trigger is pressed, the RACECAR moves backward at the designated speed.
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) == 0 \
        and rc.controller.get_trigger(rc.controller.Trigger.LEFT) == 0:
        speed = 0
    

    # TODO Part 2: Modify the following conditional statement such that when the
    # value of the left joystick's x-axis is greater than 0, the RACECAR's wheels turn right.
    # When the value of the left joystick's x-axis is less than 0, the RACECAR's wheels turn left.
    (x, y) = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    if x == 0:
        angle = 0
    elif x != 0:
        angle = x
        print(x)    
    # if x > 0.5:
    #     angle = 0.5
    # elif x < -0.5:
    #     angle = -0.5
    # else:
    #     angle = 0

    # TODO Part 3: Write a conditional statement such that when the
    # "A" button is pressed, increase the speed of the RACECAR. When the "B" button is pressed,
    # decrease the speed of the RACECAR. Print the current speed of the RACECAR to the
    # terminal window.

    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) == 1:
        speed = 1
    elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) == 1:
        speed = -1

    # if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0 or rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0:
    #     if rc.controller.was_pressed(rc.controller.Button.A):
    #         speed += speed_offset
    #     elif rc.controller.was_pressed(rc.controller.Button.B):
    #         speed -= speed_offset
    #     else:
    #         pass
    
    
    # if speed < -1 :
    #     speed = -1
    # elif speed > 1:
    #     speed = 1
    # else:
    #     pass


    # TODO Part 4: Write a conditional statement such that when the
    # "X" button is pressed, increase the turning angle of the RACECAR. When the "Y" button 
    # is pressed, decrease the turning angle of the RACECAR. Print the current turning angle 
    # of the RACECAR to the terminal window.
    if x > 0 or y > 0:
        if rc.controller.was_pressed(rc.controller.Button.X):
            angle+=angle_offset
        elif rc.controller.was_pressed(rc.controller.Button.Y):
            angle-=angle_offset
    
    angle = rc_utils.clamp(angle, -1, 1)

    # print(f"Speed: {speed}; Angle: {angle}")
    # Send the speed and angle values to the RACECAR
    rc.drive.set_speed_angle(speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
