"""
Take lidar reading, take contour reading.
    if no lidar and no contour:
        move forward slowly.
if lidar_distance < 100:
	take contour reading and selective lidar reading
if (contour is red AND selective lidar is none) OR (last state is red AND selective lidar shows object):
    set state to red
    set last state to red
else:
    set state to blue
    set last state to blue
if distance < 100 AND red:
    angle = angle + .1
elif distance < 100 AND blue:
    angle = angle - .1
"""

import sys
import cv2 as cv
import numpy as np
import time
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils
import math


rc = racecar_core.create_racecar()

MIN_CONTOUR_AREA = 2000
MIN_CONTOUR_AREA_EX = 500
MAX_CONTOUR_AREA = 20000
BLUE = ((90, 100, 100), (120, 255, 255))  # The HSV range for the color blue
RED = ((170, 100, 100), (5, 255, 255))  # The HSV range for the color red
WHITE = ((80, 44, 255), (125, 63, 255)) # HSV range for white
CROP_FLOOR = ((200, 0), (rc.camera.get_height(), rc.camera.get_width()))

STATES = {1: "RED", 2: "BLUE", 3: "FORWARD", 4: "SEARCH_B", 5: "SEARCH_R"}
set_times = []

def start():
    global current_state, last_state, speed, angle, set_time, contour_area
    current_state = "FORWARD"
    last_state = "FORWARD"
    angle = 0
    speed = 0
    set_time = time.time()
    update_contour()


def search(last_state):
    set_time = time.time()
    set_times.append(set_time)
    angle = 0
    if last_state == "RED":
        if time.time() - set_times[0] > 1:
            angle = -0.5
            last_state = "SEARCH_R"
            set_times.clear()
        else:
            angle = 0
    elif last_state == "BLUE":
        if time.time() - set_times[0] > 1:
            angle = 0.5
            last_state = "SEARCH_B"
            set_times.clear()
        else:
            angle = 0
    return angle

    

def update_contour():

    global contour_center
    global contour_area
    global current_state
    global contour_r_area, contour_b_area, lrg_contour_w, lrg_contour_b, lrg_contour_r
    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
        
        # TODO Part 2: Search for line colors, and update the global variables
        # contour_center and contour_area with the largest contour found

        # Search for contours of the current color
        contours_b = rc_utils.find_contours(image, BLUE[0], BLUE[1])
        contours_r = rc_utils.find_contours(image, RED[0], RED[1])
        contours_w = rc_utils.find_contours(image, WHITE[0], WHITE[1])
        lrg_contour_w = rc_utils.get_largest_contour(contours_w, MIN_CONTOUR_AREA)
        # print(contours_b, contours_r)
        filt_contours_b = [b for b in contours_b if cv.contourArea(b) < MAX_CONTOUR_AREA]
        filt_contours_r = [r for r in contours_r if cv.contourArea(r) < MAX_CONTOUR_AREA]
        lrg_contour_b = rc_utils.get_largest_contour(filt_contours_b, MIN_CONTOUR_AREA)
        lrg_contour_r = rc_utils.get_largest_contour(filt_contours_r, MIN_CONTOUR_AREA)

        if lrg_contour_b is not None:
            contour_b_area = rc_utils.get_contour_area(lrg_contour_b)
            contour_b_center = rc_utils.get_contour_center(lrg_contour_b)
            rc_utils.draw_circle(image, contour_b_center)
        else:
            contour_b_area = 0
            contour_b_center = None    
        if lrg_contour_r is not None:
            contour_r_area = rc_utils.get_contour_area(lrg_contour_r)
            contour_r_center = rc_utils.get_contour_center(lrg_contour_r)
            rc_utils.draw_circle(image, contour_r_center)
        else:
            contour_r_area = 0
            contour_r_center = None
        
        if contour_r_area > contour_b_area:
            contour_center = rc_utils.get_contour_center(lrg_contour_r)
            contour_area = rc_utils.get_contour_area(lrg_contour_r)
        elif contour_r_area < contour_b_area:
            contour_center = rc_utils.get_contour_center(lrg_contour_b)
            contour_area = rc_utils.get_contour_area(lrg_contour_b)
        elif contour_r_area == 0 and contour_b_area == 0:
            contour_center = None
            contour_area = 0
        rc.display.show_color_image(image)
    # print("update_contour() has been run! :)")

def update():
    global contour_r_area, contour_b_area, last_state, current_state, speed, angle, contour_area
    global should_read_contour
    scan = rc.lidar.get_samples()

    # angle_f, distance_f = rc_utils.get_lidar_closest_point(scan, (270, 90))
    # angle_l, distance_l = rc_utils.get_lidar_closest_point(scan, (225, 315))
    # angle_r, distance_r = rc_utils.get_lidar_closest_point(scan, (45, 135))
    # angle_n, distance_n = rc_utils.get_lidar_closest_point(scan, (330, 30))
    angle_l, distance_l = rc_utils.get_lidar_closest_point(scan, (270, 0))
    angle_r, distance_r = rc_utils.get_lidar_closest_point(scan, (0, 90))


    speed = 0.75
    should_read_contour = False
    # print(distance_r)
    if distance_r > 100 and angle_r == 0:
        angle = 0
        current_state = "FORWARD"
        last_state = "FORWARD"
        # angle = 0
        # angle = 0
    elif distance_r < 100 or distance_l < 100:
        if contour_r_area > contour_b_area:
            if last_state == "BLUE" and distance_r < 75:
                current_state = "BLUE"
                last_state = "BLUE"
                setpoint = 30
                current = distance_r
                kp = .003125
                error = setpoint - current
                angle = kp * error
            elif last_state == "BLUE" and distance_r > 75:
                current_state = "RED"
                last_state = "RED"
            elif last_state == "FORWARD":
                current_state = "RED"
                last_state = "RED"
                angle = .5
        elif contour_r_area < contour_b_area: 
            if last_state == "RED" and distance_l < 75:
                current_state = "RED"
                last_state = "RED"
                setpoint = 30
                current = distance_l
                kp = .003125
                error = setpoint - current
                angle = kp * error
            elif last_state == "RED" and distance_l > 75:
                current_state = "BLUE"
                last_state = "BLUE"
                angle = 0 
            elif last_state == "FORWARD":
                current_state = "BLUE"
                last_state = "BLUE"
                angle = -.5
                
        
            

    angle = rc_utils.clamp(angle, -1, 1)
    print(f"Red area: {contour_r_area}; Blue area: {contour_b_area}; Current state: {current_state}; Last state: {last_state}")
    rc.drive.set_speed_angle(speed, angle)


def update_new():
    global speed, angle, contour_r_area, contour_b_area, last_state, current_state, contour_area, lrg_contour_r, lrg_contour_b, set_time
    scan = rc.lidar.get_samples()
    angle_l, distance_l = rc_utils.get_lidar_closest_point(scan, (225, 315))
    angle_r, distance_r = rc_utils.get_lidar_closest_point(scan, (45, 135))
    contour_r_area = 0
    contour_b_area = 0
    angle = 0
    update_contour()
    
    if contour_r_area != 0 and contour_b_area == 0:
        angle = 0.5
        last_state = "RED"
    elif contour_b_area != 0 and contour_r_area == 0:
        angle = -0.5
        last_state = "BLUE"
    elif contour_r_area != 0 and contour_b_area != 0:
        r_center = rc_utils.get_contour_center(lrg_contour_r)
        b_center = rc_utils.get_contour_center(lrg_contour_b)
        if r_center[0] < b_center[0]:
            angle = 0
            last_state = "FORWARD"
        else:
            pass
    elif contour_r_area == 0 and contour_b_area == 0:
        angle = search(last_state) # ADD SEARCH HERE!!!
        
    speed = 0.75
    if rc.controller.is_down(rc.controller.Button.A):
        print(f"Left distance: {distance_l}; Right distance: {distance_r}")
        # print(f"Red area: {contour_r_area}; Blue area: {contour_b_area}")
    print(last_state)
    rc.drive.set_speed_angle(speed, angle)
    # print(f"Red area: {contour_r_area}; Blue area: {contour_b_area}")

def update_slow():
    pass

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update_new, update_slow)
    rc.go()
