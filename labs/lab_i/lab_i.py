"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_i.py

Title: Lab I - Wall Follower

Author: Kripa Sudhir << [Write your name or team name here]

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


# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils
import numpy as np

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
STATES = {1: "TURN LEFT", 2: "TURN RIGHT", 3: "MIDDLE"}
current_state = STATES[3]

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass # Remove 'pass' and write your source code for the start() function here

def calc_target(left, right):
    target = 0
    if left != 0 and right != 0:
        target = (right + left)/2.0
    elif left == 0 or right == 0:
        target = 100
    
    return target

def course_correct():
    global forward, forward_left_n, forward_right_n, forward_left_w, forward_right_w

def is_between_walls(right, left, target):                         
    return (right != target and right < left) or (left != target and right > left)

def is_one_wall(right, left, target):
    return (right != target and (right != 0 and left == 0)) or (left != target and (left != 0 and right == 0))

def is_turn_time(f_right, f_left, forward):
    return (f_right - f_left > 60 or f_left - f_right > 60) and forward < 100

def is_crashing(forward):
    return forward <= 20

def take_mean(scan):
    zero = [0]
    clean_scan = scan[~np.isin(scan, zero)]
    left = np.mean(clean_scan[560:720])
    right = np.mean(clean_scan[90:270])
    return right, left, clean_scan

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update1():
    global scan
    global to_right_90, to_left_90, forward, forward_left_n, forward_right_n, forward_left_w, \
        forward_right_w, target, mean_left, mean_right, kp, error, clean_scan, kernel
    global angle, speed, angle_r
    global current_state
    scan = rc.lidar.get_samples()

    kernel = np.array([1/3, 1/3, 1/3])
    smooth_scan = np.convolve(scan, kernel, mode='same')

    to_left_90 = scan[630] # 90 degrees to the left
    to_right_90 = scan[180] # 90 degrees to the right
    forward = scan[0]
    forward_left_w = scan[600] # 60 degrees to the left
    forward_right_w = scan[120] # 60 degrees to the right
    forward_left_n = scan[660] # 30 degrees to the left
    forward_right_n = scan[60] # 30 degrees to the right
    right_forward = rc_utils.get_lidar_average_distance(scan, 45, 20)
    left_forward = rc_utils.get_lidar_average_distance(scan, 315, 20)
    right_wall = rc_utils.get_lidar_average_distance(scan, 90, 10)
    left_wall = rc_utils.get_lidar_average_distance(scan, 270, 10)

    target = calc_target(to_right_90, to_left_90)
    # mean_right, mean_left, clean_scan = take_mean(smooth_scan)
    
    # target = (to_right_90 + to_left_90)/2.0
    # margin = 10
    kp = 0.15
    accel = rc.physics.get_linear_acceleration()
    down_accel = accel[2]
    if down_accel <= -1:
        rc.drive.set_max_speed(1)
    else:
        pass

    if is_between_walls(to_right_90, to_left_90, target):
        current_state = STATES[3]
    elif is_turn_time(forward_right_n, forward_left_n, forward):
        if forward_left_n > forward_right_n:
            current_state = STATES[1]
        elif forward_left_n < forward_right_n:
            current_state = STATES[2]



    # if is_between_walls(to_right_90, mean_left, target):
    #     current_state = STATES[3]
    # elif one_wall(mean_right, mean_left, target):
    #     if mean_right == 0:
    #         current_state = STATES[1]
    #     elif mean_left == 0:
    #         current_state = STATES[2]

    # elif turn_time(forward_right_w, forward_right_n, forward_left_w, forward_left_n, forward):
    #     if 
    
    if current_state == "TURN LEFT":
        # error = target - to_left_90
        # angle_r = kp * error
        angle = -0.5
        speed = .6
    elif current_state == "TURN RIGHT":
        # error = to_right_90 - target
        # angle_r = kp * error
        angle = 0.5
        speed = .6
    elif current_state == "MIDDLE":
        error = to_right_90 - target
        angle_r = kp * error
        speed = 0.75
    
    if is_crashing(forward):
        speed = -.5


    # if to_right_90 != target and (to_right_90 < to_left_90 or (to_right_90 != 0 and to_left_90 == 0)):
    #     error = to_right_90 - target
    #     angle_r = kp * error
    #     speed = .75
    # elif to_left_90 != target and (to_right_90 > to_left_90 or (to_left_90 != 0 and to_right_90 == 0)):
    #     error = target - to_left_90
    #     angle_r = kp * error
    #     speed = .75
    # elif forward < 100:
    #     if forward_left_n > forward_right_n:
    #         angle_r = 
    #         speed = -1
    angle = rc_utils.clamp(angle_r, -1, 1)
    # print(f"Right: {to_right_90}; Left: {to_left_90}; Forward: {forward}")
    # print(scan)

    rc.drive.set_speed_angle(speed, angle)

    
    
def update():
    global current_state, max_speed
    scan = rc.lidar.get_samples()
    right_forward = rc_utils.get_lidar_average_distance(scan, 45, 20)
    left_forward = rc_utils.get_lidar_average_distance(scan, 315, 20)
    right_wall = rc_utils.get_lidar_average_distance(scan, 90, 10)
    left_wall = rc_utils.get_lidar_average_distance(scan, 270, 10)
    
    current_state = "MIDDLE"
    kp = .15
    target = calc_target(left_wall, right_wall)

    accel = rc.physics.get_linear_acceleration()
    down_accel = accel[2]
    if down_accel <= -1:
        rc.drive.set_max_speed(1)
    else:
        pass

    if is_between_walls(right_wall, left_wall, target):
        current_state = STATES[3]
    elif is_turn_time(right_wall, left_wall, right_forward, left_forward):
        if right_forward > right_wall:
            current_state = STATES[2]
        elif left_forward > left_wall:
            current_state = STATES[1]
    # if left_forward - left_wall > 50:
    #     angle_r = -.75
    #     speed = .75
    # if right_forward - right_wall > 70:
    #     angle_r = .75
    #     speed = .75
    # else:
    #     speed = .75
    #     error = scan[180] - target
    #     kp = .15
    #     angle_r = kp * error

    if current_state == "TURN LEFT":
        # error = target - to_left_90
        # angle_r = kp * error
        angle_r = -0.5
        speed = .6
    elif current_state == "TURN RIGHT":
        # error = to_right_90 - target
        # angle_r = kp * error
        angle_r = 0.5
        speed = .6
    elif current_state == "MIDDLE":
        error = right_wall - target
        angle_r = kp * error
        speed = 0.75
    # print(f"Right wall : {right_wall}; Left wall:{left_wall}; Right forward: {right_forward}; Left forward: {left_forward}")
    angle = rc_utils.clamp(angle_r, -1, 1)

    
    rc.drive.set_speed_angle(speed, angle)

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass  # Remove 'pass and write your source code for the update_slow() function here

def update_new():
    global speed, angle
    scan = rc.lidar.get_samples()
    right_forward = rc_utils.get_lidar_average_distance(scan, 45, 20)
    left_forward = rc_utils.get_lidar_average_distance(scan, 315, 20)
    right_wall = rc_utils.get_lidar_average_distance(scan, 90, 10)
    left_wall = rc_utils.get_lidar_average_distance(scan, 270, 10)
    front_window = rc_utils.get_lidar_average_distance(scan, 0, 20)
    # print (right_forward, left_forward, right_wall, left_wall)

    accel = rc.physics.get_linear_acceleration()
    down_accel = accel[2]
    if down_accel <= -1:
        rc.drive.set_max_speed(1)
        speed = 1
    else:
        pass

    offset = right_forward - left_forward
    turn_angle = (offset % 60.0) / 60.0
    if right_forward - left_forward > 25:
        angle = turn_angle
        speed = speed - 0.1

    elif left_forward - right_forward > 25:
        angle = -turn_angle
        speed = speed - 0.1
    else:
        angle = 0
        speed = 1 

    if front_window < 110:
        if angle <= 0:
            angle = angle - 0.5
        else:
            angle = angle + 0.5

    if speed < 0.5:
        speed = 0.7

    angle_n = rc_utils.clamp(angle, -1, 1)
    rc.drive.set_speed_angle(speed, angle_n)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update_new, update_slow)
    rc.go()
