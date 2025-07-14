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

MIN_CONTOUR_AREA = 1500
MIN_CONTOUR_AREA_EX = 500
MAX_CONTOUR_AREA = 20000
BLUE = ((90, 100, 100), (120, 255, 255))  # The HSV range for the color blue
RED = ((170, 100, 100), (5, 255, 255))  # The HSV range for the color red
WHITE = ((80, 44, 255), (125, 63, 255)) # HSV range for white
CROP_FLOOR = ((200, 0), (rc.camera.get_height(), rc.camera.get_width()))

STATES = {1: "RIGHT", 2: "LEFT", 3: "FORWARD"}


def start():
    global current_state, last_state, speed, angle, set_time, contour_area
    current_state = "FORWARD"
    last_state = "FORWARD"
    angle = 0
    speed = 0
    set_time = time.time()
    update_contour()


def update_contour():

    global contour_center
    global contour_area
    global current_state
    global contour_r_area, contour_b_area, lrg_contour_w
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
        if lrg_contour_w is None:
            lrg_contour_b = rc_utils.get_largest_contour(filt_contours_b, MIN_CONTOUR_AREA)
            lrg_contour_r = rc_utils.get_largest_contour(filt_contours_r, MIN_CONTOUR_AREA)
        else:
            lrg_contour_b = rc_utils.get_largest_contour(filt_contours_b, MIN_CONTOUR_AREA_EX)
            lrg_contour_r = rc_utils.get_largest_contour(filt_contours_r, MIN_CONTOUR_AREA_EX)

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

def update():
    global contour_r_area, contour_b_area, last_state, current_state, speed, angle, contour_area
    global should_read_contour
    scan = rc.lidar.get_samples()

    angle_f, distance_f = rc_utils.get_lidar_closest_point(scan, (270, 90))
    angle_l, distance_l = rc_utils.get_lidar_closest_point(scan, (225, 315))
    angle_r, distance_r = rc_utils.get_lidar_closest_point(scan, (45, 135))
    angle_n, distance_n = rc_utils.get_lidar_closest_point(scan, (330, 30))

    should_read_contour = False

    if distance_f > 100:
        speed = 0.75
        # angle = 0
        # angle = 0
        if current_state == 'RED':
            angle = -0.75
        elif current_state == 'BLUE':
            angle = 0.75
        else:
            current_state = "FORWARD"
            angle = 0

        if not should_read_contour:
            update_contour()
            should_read_contour = False
    elif distance_f <= 100:
        if should_read_contour:
            update_contour()
            should_read_contour = False

        if contour_r_area > contour_b_area:
            current_state = "RED"
            # if angle_l > 250 and distance_f < 150:
            #     angle = -0.75
            #     speed = 0.5
            # elif angle_l < 250 and distance_f < 100:
            #     angle = 0.75
            #     speed = 0.5
            # elif distance_f > 120:
            #     should_read_contour = True
            
            if distance_f < 100:
                angle = 0.75
                speed = 0.5
            elif distance_f < 120 or angle_l > 350:
                angle = -0.75
                speed = 0.5
            else:
                should_read_contour = True

        elif contour_r_area < contour_b_area:
            current_state = "BLUE"
            if distance_f < 100:
                angle = -1
                speed = 0.5
            elif distance_f < 120 or angle_r > 10:
                angle = 1
                speed = 0.5
            else:
                should_read_contour = True

    angle = rc_utils.clamp(angle, -1, 1)
    print(f"State: {current_state}, angle_l: {angle_l}, angle: {angle}, distance_l: {distance_l}")
    rc.drive.set_speed_angle(speed, angle)
    

def update_slow():
    pass

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
