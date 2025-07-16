"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: grand_prix.py

Title: Grand Prix Day!

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: Write a script to enable fully autonomous behavior from the RACECAR. The
RACECAR will traverse the obstacle course autonomously without human intervention.
Once the start button is pressed, the RACECAR must drive through the course until it
reaches finish line.

Note: There is no template code in this document to follow except for the RACECAR script 
structure found in template.py. You are expected to use code written from previous labs
to complete this challenge. Good luck!

Expected Outcome: When the user runs the script, they must not be able to manually control
the RACECAR. The RACECAR must move forward on its own, traverse through the course, and then
stop on its own.
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

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
import cv2
import math

# Declare any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

#for wall detection
walls_detected = ""
queue = [] # The queue of instructions
state = ""
stay_dist = 50


# Load dictionary and parameters from the aruco library
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
AR_detected = 0
markercount = 0
markertimer = 0

# HSV Thresholds + color detector
ORANGE = ((10,170,175),(20,255,255)) # The HSV range for the color orange
YELLOW = ((25,100,175),(35,255,255)) # The HSV range for the color yellow
#PURPLE = ((115,50,175),(150,255,255)) # The HSV range for the color purple
BLUE = ((90, 150, 150),(120, 255, 255), "BLUE")
GREEN = ((35, 50, 50),(85, 255, 255), "GREEN")
RED = ((170, 115, 115),(10, 255, 255), "RED")
COLORS = [BLUE, GREEN, RED, ORANGE, YELLOW] # List of colors
COLOR_PRIORITY =  (RED, GREEN, BLUE)

MIN_CONTOUR_AREA = 30

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()-100))

contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

#other 

########################################################################################
# Functions
########################################################################################


# [CLASS] AR Markers
class ARMarker:
    def __init__(self, marker_id, marker_corners, orientation, area):
        self.id = marker_id
        self.corners = marker_corners
        self.orientation = orientation # Orientation of the marker
        self.area = area # Area of the marker
        self.color = ""
        self.color_area = 0

    def find_color_border(self, image):
        # Find the crop points and slice the image to the AR Marker
        crop_points = self.find_crop_points(image)
        image = image[crop_points[0][0]:crop_points[0][1], crop_points[1][0]:crop_points[1][1]]
        
        # Find the colors from the image
        color_name, color_area = self.find_colors(image)
        self.color = color_name
        self.color_area = color_area

    # [FUNCTION] Find the crop points for the AR Marker
    def find_crop_points(self, image):
        ORIENT = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3}
        current_orientation = self.orientation
        
        # Top left corner mappings: UP = 0, LEFT = 1, DOWN = 2, RIGHT = 3
        marker_left, marker_top = self.corners[ORIENT[current_orientation]] # marker.corners are (x, y) -> (col, row) -> (right/left, top/down)
        # Bottom right corner mappings: UP = 3, LEFT = 4, DOWN = 1, RIGHT = 2
        marker_right, marker_bottom = self.corners[(ORIENT[current_orientation] + 2) % 4] # marker.corners are (x, y) -> (col, row)

        # Add half of marker length and marker width to crop points
        half_marker_length = (marker_right - marker_left) // 2
        half_marker_width = (marker_bottom - marker_top) // 2
        
        marker_top = max(0, marker_top - half_marker_width) # max function prevents value from decreasing past 0
        marker_left = max(0, marker_left - half_marker_length)
        marker_bottom = min(image.shape[0], marker_bottom + half_marker_width) + 1 # +1 prevents value from increasing past frame limits
        marker_right = min(image.shape[1], marker_right + half_marker_length) + 1

        return ((int(marker_top), int(marker_bottom)), (int(marker_left), int(marker_right)))

    # [FUNCTION] Find the colors in the image
    def find_colors(self, image):
        color_name = "None" # The detected color from the list of color thresholds
        color_area = 0 # The area of the detected color
        for (hsv_lower, hsv_upper, color) in COLORS:
            contours = rc_utils.find_contours(image, hsv_lower, hsv_upper)
            largest_contour = rc_utils.get_largest_contour(contours)
            if largest_contour is not None:
                contour_area = rc_utils.get_contour_area(largest_contour)
                if contour_area > color_area:
                    color_area = contour_area
                    color_name = color

        return color_name, color_area


# [FUNCTION] Detect AR Tag from image and return list of detections
def detect_AR_Tag(image):
    # Output: A list of detected AR Markers (returns as empty if none)
    global markers, area
    markers = []
    area = 0
    

    # Detect AR Marker from image and return their corners and IDs
    corners, ids, _ = detector.detectMarkers(image)
    
    # Loop through all corner and ID indexes
    for x in range(len(corners)):
        # Retrieve current corner
        current_corners = corners[x][0]

        # Corners is a list of four elements: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # The first element is the origin, the next elements are corners clockwise
        orientation = ""
        if current_corners[0][0] == current_corners[1][0]: # if x1 = x2, RIGHT or LEFT
            if current_corners[0][1] > current_corners[1][1]: # if y1 > y2, LEFT
                orientation = "LEFT"
            else: # if y2 > y1, RIGHT
                orientation = "RIGHT"
        else: # if x1 != x2, UP or DOWN
            if current_corners[0][0] > current_corners[1][0]: # if x1 > x2, DOWN
                orientation = "DOWN"
            else: # if x2 > x1, UP
                orientation = "UP"
            
        # Find the area of each marker
        area = abs((current_corners[2][0] - current_corners[0][0])) * abs((current_corners[2][1] - current_corners[0][1]))

        # Create current marker object and add to list
        current_marker = ARMarker(ids[x][0], current_corners, orientation, area)
        markers.append(current_marker)

    cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))

    return markers, image

def updateStateAndMarker():
    global markercount 
    global current_state
    global state
    global markertimer 

    image = rc.camera.get_color_image()
    corners, ids, _ = detector.detectMarkers(image)
    for x in range(len(corners)): # will not run if no corners are detected!
        curr_corners = corners[x][0]
        area = abs((curr_corners[2][0] - curr_corners[0][0])) * abs((curr_corners[2][1] - curr_corners[0][1]))
        print(area)
        if area > 1200 and markertimer > 5.0:
            markercount +=1
            markertimer = 0.0
    markertimer += rc.get_delta_time()

    
    #updating state
    if(markercount == 0 or markercount == 2 or markercount == 5): current_state = wall_follower(), print("walls")
    if(markercount == 1): current_state = between_colors(), print("betw")
    if(markercount == 3): current_state = cone_slalom(), print("cones")
    if(markercount == 4): current_state = color_follower(), print("follow") 
   

def wall_follower():
    global speed
    global angle, state
    global check_right
    stay_dist = 50
    # Access the most recent lidar scan.
    scan = rc.lidar.get_samples()

    # Get the distance of the measurement directly in front of the car
    _, forward_distance = rc_utils.get_lidar_closest_point(scan, (335, 7.5))
    check_right = rc_utils.get_lidar_average_distance(scan, 90, 90)
    check_left = rc_utils.get_lidar_average_distance(scan, 270, 90)

    max_wall_dist = 170

    right_wall = check_right < max_wall_dist
    left_wall = check_left < max_wall_dist

    if right_wall and left_wall:
        state = "center"
    elif right_wall:
        state = "right"
    elif left_wall:
        state = "left"
    else:
        state = "no_walls"
        error = 0
        speed = 0.3

    if state == "center":
        error = check_right - check_left

    elif state == "right":
        if check_right < 45: #if really close, keep stay distance smaller for tight turns
            stay_dist = 50
        else:
            stay_dist = 25
        error = stay_dist - check_right

    elif state == "left":
        if check_left < 45:
            stay_dist = 50
        else:
            stay_dist = 25
        error = stay_dist - check_left

    elif state == "no_walls":
        error = stay_dist

    #control
    if abs(error) > 30:
        kp_a = 0.3
    else: 
        kp_a = 0.01
    angle = kp_a * (error)
    angle = max(min(angle,1),-1)

    kp_s = 0.008
    raw_speed = kp_s * forward_distance
    if forward_distance > 40:
        speed = max(min(raw_speed,1),0.1)
    else:
        speed = 0.1
    
    if abs(angle) > 0.7:
        speed *= 0.85

    rc.drive.set_speed_angle(speed, angle)

def update_contour():
    global contour_center
    global contour_area
    global colorset
    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
        print("contour is none")
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # TODO Part 2: Search for line colorscd, and update the global variables
        # contour_center and contour_area with the largest contour found
        
        for color in COLORS:
            #print(f"COLOR {color[0][0]}")
            contours = rc_utils.find_contours(hsv,color[0],color[1])
            
            contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA) #select largest contour
            if contour is not None:
                contour_center = rc_utils.get_contour_center(contour)
                contour_area= rc_utils.get_contour_area(contour)  
                rc_utils.draw_contour(image,contour)
            
           # rc_utils.draw_cirlce(image,contour_center)
        # Display the image to the screen
            rc.display.show_color_image(image)

def between_colors():
    global angle 
    update_contour()
    if contour_center is not None:
        setpoint = rc.camera.get_width() //3
        present_value = contour_center[1]
        kp = -0.003125
        error = setpoint - present_value
        angle = kp*error
        angle = rc_utils.clamp(angle,-1,1)
    

    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    rc.drive.set_speed_angle(speed, angle)

def color_follower():
    global angle, speed
    if contour_center is not None:
        setpoint = rc.camera.get_width() //2
        present_value = contour_center[1]
        kp = -0.003125
        error = setpoint - present_value
        angle = kp*error
        rc_utils.clamp(angle,-1,1)

    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    rc.drive.set_speed_angle(speed, angle)

def cone_slalom():
    pass

def clamp(value:float, min: float, max: float):
    if value < min:
        return min
    elif value > max:
        return max
    else: 
        return value

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, max_speed
    global angle
    global AR_detected
    AR_detected = 0
    # Initialize variables
    speed = 0.5
    max_speed = 0.09
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle) 

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global image
    global markers, area, AR_detected
    global speed, max_speed, angle, old_angle 
    global state

    image = rc.camera.get_color_image()
    detect_AR_Tag(image)
    update_contour()
    updateStateAndMarker()
    rc.drive.set_speed_angle(speed, angle) 

    '''
    if len(markers) > AR_detected and area > 4000:
        AR_detected +=1

    if AR_detected == 0:
        wall_follower () 
    
    elif AR_detected == 1:
        update_contour()
        color_follower()

    elif AR_detected == 2:
        wall_follower()
    
    elif AR_detected == 3:
        cone_slalom()

    elif AR_detected == 4:
        color_follower()
    
    elif AR_detected == 5:
        wall_follower()
    
    else:
        print("hmmm fix ur code")
    '''

    

    
    #rc.display.show_color_image(image)
    

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    #print (AR_detected) # Remove 'pass and write your source code for the update_slow() function here
    #print(state)
    #if area >0:
       # print(area)
    pass

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
