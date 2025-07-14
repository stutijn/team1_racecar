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

STATES = {1: "RIGHT", 2: "LEFT", 3: "SEARCH", 4: "FORWARD"}
global current_state
current_state = ""




# update contour function
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
        rc.display.show_color_image(image)