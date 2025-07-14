import sys

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils
import cv2
import numpy as np

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

ORANGE = ((10, 50, 50), (20, 255, 255)) # HSV range for orange
PURPLE = ((130, 50, 50), (160, 255, 255)) # HSV range for purple
MIN_CONTOUR_AREA = 50
CROP_LEFT = ((2 * rc.camera.get_height()//3, 0), (rc.camera.get_height(), rc.camera.get_width()//3))
CROP_RIGHT = ((2 * rc.camera.get_height()//3 - 100, 2 * rc.camera.get_height() //3), (rc.camera.get_height(), rc.camera.get_width()))

def start():
    pass

def update_contour_between(image):
    image = rc.camera.get_color_image()


    if image is None:
        contour_center_1 = None
        contour_area_1 = 0
        contour_center_2 = None
        contour_area_2 = 0
        contour_center_3 = None
        contour_area_3 = 0
    else:
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
        contours_o = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        if contours_o is not None and len(contours_o) >= 2:
            sorted_o = sorted(contours_o, key=cv2.contourArea, reverse=True)
            lrg_contour_o1 = rc_utils.get_largest_contour(sorted_o[0], MIN_CONTOUR_AREA)
            lrg_contour_o2 = rc_utils.get_largest_contour(sorted_o[1], MIN_CONTOUR_AREA)

            center_o1 = rc_utils.get_contour_center(lrg_contour_o1)
            if center_o1 is not None:
                rc_utils.draw_circle(image, center_o1)
            else:
                pass
            center_o2 = rc_utils.get_contour_center(lrg_contour_o2)
            if center_o2 is not None:
                rc_utils.draw_circle(image, center_o2)
            else:
                pass
        else:
            pass

        contours_p = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])
        if contours_p is not None and len(contours_p) >= 2:
            sorted_p = sorted(contours_p, key=cv2.contourArea, reverse=True)
            lrg_contour_p1 = rc_utils.get_largest_contour(sorted_p[0], MIN_CONTOUR_AREA)
            lrg_contour_p2 = rc_utils.get_largest_contour(sorted_p[1], MIN_CONTOUR_AREA)

            center_p1 = rc_utils.get_contour_center(lrg_contour_p1)
            print(f"Contours: {contours_p}, Large: {lrg_contour_p1}")

            if center_p1 is not None:
                rc_utils.draw_circle(image, center_p1)

            else:
                pass
            center_p2 = rc_utils.get_contour_center(lrg_contour_p2)
            if center_p2 is not None:
                rc_utils.draw_circle(image, center_p2)
            else:
                pass
        else:
            pass

        # Display the image to the screen
        rc.display.show_color_image(image)

def update_contour_test():
    image = rc.camera.get_color_image()
    image_l = rc_utils.crop(image, CROP_LEFT[0], CROP_LEFT[1])
    image_r = rc_utils.crop(image, CROP_RIGHT[0], CROP_RIGHT[1])
    if image is not None:
        contours_o = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contours_p = rc_utils.find_contours(image, PURPLE[0], PURPLE[1]) 
        if contours_o is not None:
            lrg_contour_o = rc_utils.get_largest_contour(contours_o, MIN_CONTOUR_AREA)
            contour_center = rc_utils.get_contour_center(lrg_contour_o)
            rc_utils.draw_circle(image, contour_center)   
        elif contours_p is not None:
            lrg_contour_p = rc_utils.get_largest_contour(contours_p, MIN_CONTOUR_AREA)
            contour_center = rc_utils.get_contour_center(lrg_contour_p)  
            rc_utils.draw_circle(image, contour_center)     
    
    rc.display.show_color_image(image_r)
    
    if contour_center is not None:
        setpoint = rc.camera.get_width() // 2 + 200
        present_value = contour_center[1]
        kp = 0.003125
        error = present_value - setpoint # setpoint - present_value
        angle = kp * error
        angle = rc_utils.clamp(angle, -1, 1)
    elif contour_center is None:
        angle = 0
        print("No contour found")
    speed = 0.5
    # rc.display.show_color_image(image_r)
    rc.drive.set_speed_angle(speed, angle)
    

def update():
    image = rc.camera.get_color_image()
    update_contour_between(image)

def update_slow():
    image = rc.camera.get_color_image()

    # if image is not None:
    #     contours_p = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])

    #     if contours_p is not None:
    #         largest = rc_utils.get_largest_contour(contours_p, MIN_CONTOUR_AREA)
    #         print(contours_p, type(largest))
    #         list(contours_p).remove(largest)
    #         second_largest = rc_utils.get_largest_contour(contours_p, MIN_CONTOUR_AREA)
    #         center1 = rc_utils.get_contour_center(largest)
    #         center2 = rc_utils.get_contour_center(second_largest)
    #         print("Center1:", center1, "Contour2:", center2)
    #         rc_utils.draw_circle(image, center1)
    #         rc.display.show_color_image(image)
    #         rc_utils.draw_circle(image, center2)
    #         rc.display.show_color_image(image)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update_contour_test, update_slow)
    rc.go()