"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [PLACEHOLDER] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core
import cv2
import racecar_utils as rc_utils
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
# Declare any global variables here
BLUE = ((90, 115, 115), (120, 255, 255)) # HSV range for blue
GREEN = ((40, 115, 115), (80, 255, 255)) # HSV range for green
RED = ((0, 115, 115), (10, 255, 255)) # HSV range for red

########################################################################################
# Functions
########################################################################################

class ArUco(object):
    def __init__(self, ids, corners, orientation, area):
        self.ids = ids
        self.corners = corners
        self.orientation = orientation   # orientation of the marker
        self.area = area  # area of the marker

    


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass
     # Remove 'pass' and write your source code for the start() function here

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def find_orientation(corners):
    if corners[0][0] == corners[1][0]: # x1 = x2, so it must be RIGHT or LEFT
        if corners[0][1] > corners[1][1]: # y1 > y2
            return "LEFT"
        elif corners[0][1] < corners[1][1]: # y2 > y1
            return "RIGHT"
    elif corners[0][0] != corners[1][0]: # x1 =/= x2, must be DOWN or UP
        if corners[0][0] > corners[1][0]: # x1 > x2
            return "DOWN"
        elif corners[0][0] < corners[1][0]: # x1 < x2
            return "UP"


def detect_AR_Tag(image):
    global contour_center, contour_area, markers
    markers = []
    corners, ids, _ = detector.detectMarkers(image)
     

    for x in range(len(corners)):
        current_corner = corners[x][0]

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
        
        print(f"ID: {ids[0]} || Corners: {corners[0]}")
        return markers, image

"""if current_corner[]"""

def update():
    image = rc.camera.get_color_image()
    results = detect_AR_Tag(image)
    if results is not None:
        markers, image = results
        rc.display.show_color_image(image)

        print(f"======== Detection Summary ========")
        print(f"Amount of AR Tags Found: {len(markers)}")
        for marker in markers:
            print(f"Marker ID: {marker.ids} || Marker Orientation: {marker.orientation} || Marker Area: {marker.area}")
            
        
        print(f"======== End of Summary ========\n")
        # corners, ids = detect_AR_Tag(image)
        # cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
        rc.display.show_color_image(image)
    else:
        print(f"No AR marker detected") 

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    # image = rc.camera.get_color_image()
    # rc.display.show_color_image(image) 
    pass # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
