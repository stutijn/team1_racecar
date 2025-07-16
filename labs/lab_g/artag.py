########################################################################################
# Imports
########################################################################################

import sys
import cv2

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

import cv2

# Load dictionary and parameters from the aruco library
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# HSV Thresholds
BLUE = ((90, 115, 115),(120, 255, 255), "BLUE")
GREEN = ((40, 115, 115),(80, 255, 255), "GREEN")
RED = ((170, 115, 115),(10, 255, 255), "RED")
COLORS = [BLUE, GREEN, RED] # List of colors

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
    markers = []

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

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass # Remove 'pass' and write your source code for the start() function here

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global markers
    image = rc.camera.get_color_image()

    #Find all AR Markers
    markers = rc_utils.get_ar_markers(image) #object contains ID, corners, color, and color area params
    #Draw AR Markers
    rc_utils.draw_ar_markers(image,markers)
    for marker in markers:
        marker.detect_colors(image, [BLUE, RED, GREEN])

        
  # markers, image = detect_AR_Tag(image) #hashed used for OpenCV library
    
    # Crop out the AR marker with ID = 3
 #  marker_of_interest = 3
  # for marker in markers:
   #    if marker.id == marker_of_interest:
    #       marker.find_color_border(image)
     #      break
    
    rc.display.show_color_image(image)

    # Print the corners and id of the first detected marker to the terminal
    


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    print("\n\n==================================")
    for marker in markers:
        print(f"Marker ID: {marker.get_id()}")
        print(f"Marker Corners: {marker.get_corners_aruco_format()}")
        print(f"Marker Color: {marker.get_color()}")
        print(f"Marker Orientation: {marker.get_orientation()}")
        print("==================================")

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()