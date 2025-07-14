
########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv #type: ignore
import numpy as np #type: ignore

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, "../../library")
import racecar_core#type: ignore
import racecar_utils as rc_utils#type: ignore


########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# TODO Part 1: Determine the HSV color threshold pairs for GREEN and RED
# Colors, stored as a pair (hsv_min, hsv_max) Hint: Lab E!
BLUE = ((90, 100, 100), (120, 255, 255), 'blue')  # The HSV range for the color blue
GREEN = ((40, 75, 75), (70, 255, 255), "green")  # The HSV range for the color green
RED1 = ((170, 100, 100), (180, 255, 255), "red")  # The HSV range for the color red
RED2 = ((0, 100, 100), (10, 255, 255), "red")
ORANGE = ((10, 100, 100), (20, 255, 255), "orange") # The HSV range for the color orange
YELLOW = ((22, 100, 100), (28, 255, 255), "yellow") # The HSV range for the color yellow
PURPLE = ((130, 100, 100), (160, 255, 255), "purple") # The HSV range for the color purple

# Color priority: Red >> Green >> Blue
COLOR_PRIORITY = (RED1, RED2, GREEN, ORANGE, YELLOW, PURPLE, BLUE)

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
wall_color=''


########################################################################################
# Functions
########################################################################################

#Sort through the colors and find the line to follow
def getLineColor(color):
    image=rc.camera.get_color_image()
    image=rc_utils.crop(image, (300, 0), (rc.camera.get_height(), rc.camera.get_width()))

    hsvLower=color[0]
    hsvUpper=color[1]

    hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv, hsvLower,hsvUpper)

    contours, _=cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    if (len(contours)>0):
        maxContour=[contours[0]]
        for contour in contours:
            if (cv.contourArea(contour)>50 and cv.contourArea(contour)>cv.contourArea(maxContour[0])):
                maxContour[0]=contour

        return [cv.contourArea(maxContour[0]), rc_utils.get_contour_center(maxContour[0]), maxContour[0]]
    else:
        return [-1]
#Sort through colors to see if there's a wall in the way
def getMaxContour(color):
    image=rc.camera.get_color_image()
    image=rc_utils.crop(image, (175, 0), (rc.camera.get_height(), rc.camera.get_width()))

    hsvLower=color[0]
    hsvUpper=color[1]

    hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv, hsvLower,hsvUpper)

    contours, _=cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    if (len(contours)>0):
        maxContour=[contours[0]]
        for contour in contours:
            if (cv.contourArea(contour)>cv.contourArea(maxContour[0])):
                maxContour[0]=contour
        

        return [cv.contourArea(maxContour[0]), rc_utils.get_contour_center(maxContour[0]), maxContour[0]]
    else:
        return [-1]
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    #Racecar max speed, can be changed
    rc.drive.set_max_speed(.91) 
    global speed
    global angle
    global contour_center
    global contour_area
    global error
    global lineColor

    #Look for the line to follow
    lineColor=False
    for color in COLOR_PRIORITY:
        if getLineColor(color)[0]!=-1:
            lineColor=color
            break
    if lineColor:
        contour_area=getLineColor(lineColor)[0]
        contour_center=getLineColor(lineColor)[1]
        image=rc.camera.get_color_image()
        image=rc_utils.crop(image, (300, 0), (rc.camera.get_height(), rc.camera.get_width()))
        cv.drawContours(image, getLineColor(lineColor)[2], -1, (0, 255, 0), 3)
        rc.display.show_color_image(image)

  
    if contour_center is not None:
        #Find setpoint and current conoutr center
        setpoint=rc.camera.get_width()//2
        presentVal=contour_center[1]

        #Calculate angle
        kp=-.005
        error=setpoint-presentVal
        unclamped=kp*error

        #Clamp the angle
        angle=rc_utils.clamp(unclamped, -1, 1)

    global wall_color
    maxColor=COLOR_PRIORITY[0]
    for color in COLOR_PRIORITY:
        if getMaxContour(color)[0]>getMaxContour(maxColor)[0]:
            maxColor=color

    if (len(getMaxContour(maxColor))>1):
        contour_center=getMaxContour(maxColor)[1]
        contour_area=getMaxContour(maxColor)[0]
        wall_color=maxColor[2]
    
    if wall_color!="purple" and wall_color!="orange":
        speed=1
    else:
        speed=0

    rc.drive.set_speed_angle(rc_utils.clamp(speed-0.3, 0, 1), angle)

    

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the contour area and x-position
    if rc.camera.get_color_image() is None:
        # If no image is found, print all X's and don't display an image
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If an image is found but no contour is found, print all dashes
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area))

        # Otherwise, print a line of dashes with a | indicating the contour x-position
        else:
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
