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
import cv2
import time

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
# Declare any global variables here
STATES = {1: "TURN LEFT", 2: "TURN RIGHT", 3: "MIDDLE"}
current_state = STATES[3]

# MIN_CONTOUR_AREA = 30
CROP_FLOOR = ((200, 0), (rc.camera.get_height(), rc.camera.get_width()))
MIN_CONTOUR_AREA = 500
CROP_LEFT = ((2 * rc.camera.get_height()//3 - 50, 0), (rc.camera.get_height(), rc.camera.get_height() //3))

BLUE = ((90, 50, 50), (120, 255, 255))  # The HSV range for the color blue
GREEN = ((35, 50, 50), (75, 255, 255))  # The HSV range for the color green
RED = ((170, 50, 50), (5, 255, 255))  # The HSV range for the color red
ORANGE = ((10, 50, 50), (20, 255, 255)) # HSV range for orange
PURPLE = ((130, 50, 50), (160, 255, 255)) # HSV range for purple

queue = []

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global marker_count, marker_found
    marker_count = 0
    marker_found = False

def walls(right_f, left_f, front_w):
    speed = 0
    angle = 0
    offset = right_f - left_f
    turn_angle = (offset % 60.0) / 60.0
    if right_f - left_f > 25:
        angle = turn_angle
        speed = speed - 0.1
    elif left_f - right_f > 25:
        angle = -turn_angle
        speed = speed - 0.1
    else:
        angle = 0
        speed = 1 

    if front_w < 110:
        if angle <= 0:
            angle = angle - 0.5
        else:
            angle = angle + 0.5
    if speed < 0.5:
        speed = 0.7
    return angle, speed
# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  

def stay_between_lines(right, left):
    present_value = (right - left) / 2
    setpoint = rc.camera.get_width() // 2
    kp = 0.003125
    error = present_value - setpoint # setpoint - present_value
    angle = kp * error
    angle = rc_utils.clamp(angle, -1, 1)
    return angle


class ArUco(object):
    def __init__(self, ids, corners, orientation, area):
        self.ids = ids
        self.corners = corners
        self.orientation = orientation   # orientation of the marker
        self.area = area  # area of the marker

        
def update_contour_between():
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()
    
    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_LEFT[0], CROP_LEFT[1])

        # TODO Part 2: Search for line colors, and update the global variables
        # contour_center and contour_area with the largest contour found

        # Search for contours of the current color
        contours_o = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contours_p = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])
        # contours_g = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        lrg_contour_o = rc_utils.get_largest_contour(contours_o, MIN_CONTOUR_AREA)
        if lrg_contour_o is not None:
            area_o = rc_utils.get_contour_area(lrg_contour_o)
        else:
            area_o = 0
        lrg_contour_p = rc_utils.get_largest_contour(contours_p, MIN_CONTOUR_AREA)
        if lrg_contour_p is not None:
            area_p = rc_utils.get_contour_area(lrg_contour_p)
        else:
            area_p = 0
        # lrg_contour_g = rc_utils.get_largest_contour(contours_g, MIN_CONTOUR_AREA)

        if lrg_contour_o is not None and area_o > area_p:
            contour_center = rc_utils.get_contour_center(lrg_contour_o)
            contour_area = rc_utils.get_contour_area(lrg_contour_o)
            rc_utils.draw_circle(image, contour_center)
        if lrg_contour_p is not None and area_p > area_o:
            contour_center = rc_utils.get_contour_center(lrg_contour_p)
            contour_area = rc_utils.get_contour_area(lrg_contour_p)
            rc_utils.draw_circle(image, contour_center)
        elif lrg_contour_p is None and lrg_contour_o is None:
            contour_center = None
            contour_area = 0
        # elif lrg_contour_b is not None:
        #     contour_center = rc_utils.get_contour_center(lrg_contour_b)
        #     contour_area = rc_utils.get_contour_area(lrg_contour_b)
        
        # Display the image to the screen
        rc.display.show_color_image(image)

def update_between():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    accel = rc.physics.get_linear_acceleration()
    down_accel = accel[2]
    speed = 0.75
    update_contour_between()
    if contour_center is not None and contour_center[1] > 50 and contour_center[0] < 130:
        setpoint = 90
        present_value = contour_center[1]
        kp = -0.003125
        error = present_value - setpoint # setpoint - present_value
        angle = kp * error
        angle = rc_utils.clamp(angle, -1, 1)
    elif contour_center is None:
        angle = -1
        print("No contour detected")
    elif contour_center[0] > 130: # TODO: was 100
        angle = 1
        
    # elif contour_center[0] > 100 and down_accel > 1.5:
    #     angle = -1
    elif down_accel < -1.5:
        # angle = -.5
        speed = 1
        
        
    # print(contour_area)
    # if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0:
    #     speed = .5
    # elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0:
    #     speed = -.5
    print(contour_area)
    return angle, speed



def detect_AR_Tag(image):
    global area, markers, corners, ids, orientation, marker_count, time_set
    markers = []
    corners, ids, _ = detector.detectMarkers(image)
     

    for x in range(len(corners)):
        current_corner = corners[x][0]
        time_set = time.time()

        orientation = ""

        if current_corner[0][0] == current_corner[1][0]: # x1 = x2, so it must be RIGHT or LEFT
            if current_corner[0][1] > current_corner[1][1]: # y1 > y2
                orientation = "LEFT"
            elif current_corner[0][1] < current_corner[1][1]: # y2 > y1
                orientation = "RIGHT"
        elif current_corner[0][0] != current_corner[1][0]: # x1 =/= x2, must be DOWN or UP
            if current_corner[0][0] > current_corner[1][0]: # x1 > x2
                orientation = "DOWN"
            elif current_corner[0][0] < current_corner[1][0]: # x1 < x2
                orientation = "UP"
            
        area = (current_corner[2][0] - current_corner[0][0]) * (current_corner[2][1] - current_corner[0][1])
        current_marker = ArUco(ids[x][0], corners[x][0], orientation, abs(area))
        markers.append(current_marker)

        cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
        
        # print(f"ID: {ids[0]} || Corners: {corners[0]}")
        return markers, image
    

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass  # Remove 'pass and write your source code for the update_slow() function here

def update_new():
    global speed, angle, markers, marker_count, marker_found, queue
    scan = rc.lidar.get_samples()
    right_forward = rc_utils.get_lidar_average_distance(scan, 45, 20)
    left_forward = rc_utils.get_lidar_average_distance(scan, 315, 20)
    right_wall = rc_utils.get_lidar_average_distance(scan, 90, 10)
    left_wall = rc_utils.get_lidar_average_distance(scan, 270, 10)
    front_window = rc_utils.get_lidar_average_distance(scan, 0, 20)
    image = rc.camera.get_color_image()

    if len(queue) > 0:
        speed = queue[0][1]
        angle = queue[0][2]
        queue[0][0] -= rc.get_delta_time()
        if queue[0][0] <= 0:
            queue.pop(0)
    
    if detect_AR_Tag(image) is not None and area > 2000:
        if not marker_found:
            marker_count += 1
            marker_found = True
        else:
            pass
    else:
        marker_found = False


    if marker_count == 0:
        angle, speed = walls(right_forward, left_forward, front_window)
    elif marker_count == 1:
        angle, speed = update_between()
    elif marker_count == 2:
        queue.append([1.5, 1, 0])
        angle, speed = walls(right_forward, left_forward, front_window)
    elif marker_count == 3:
        pass
        

    # print(marker_count)
    angle_n = rc_utils.clamp(angle, -1, 1)
    rc.drive.set_speed_angle(speed, angle_n)

def update_test():
    image = rc.camera.get_color_image()
    detect_AR_Tag(image)
    print(marker_count)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update_new, update_slow)
    rc.go()
