"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: AR_Fork.py

Title: AR Fork

Author: TEAM 1

Purpose: As the car approaches the dynamic fork obstacle, it should determine the orientation of the AR tag mounted on the fork, and turn appropriately. 

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
########################################################################################
# Global variables
########################################################################################
marker = None
markers = None
rc = racecar_core.create_racecar()
count = 0

    
class ARMarker:
    def __init__(self, marker_id, marker_corners, orientation, area):
        self.id = marker_id
        self.corners = marker_corners
        self.orientation = orientation 
        self.area = area 
        self.color = ""
        self.color_area = 0

    def find_color_border(self, image):
    
        crop_points = self.find_crop_points(image)
        image = image[crop_points[0][0]:crop_points[0][1], crop_points[1][0]:crop_points[1][1]]
        

        color_name, color_area = self.find_colors(image)
        self.color = color_name
        self.color_area = color_area


    def find_crop_points(self, image):
        ORIENT = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3}
        current_orientation = self.orientation
        
        marker_left, marker_top = self.corners[ORIENT[current_orientation]] 
        marker_right, marker_bottom = self.corners[(ORIENT[current_orientation] + 2) % 4]

        half_marker_length = (marker_right - marker_left) // 2
        half_marker_width = (marker_bottom - marker_top) // 2
        
        marker_top = max(0, marker_top - half_marker_width) 
        marker_left = max(0, marker_left - half_marker_length)
        marker_bottom = min(image.shape[0], marker_bottom + half_marker_width) + 1 
        marker_right = min(image.shape[1], marker_right + half_marker_length) + 1

        return ((int(marker_top), int(marker_bottom)), (int(marker_left), int(marker_right)))
    

    def find_colors(self, image):
        color_name = "None" 
        color_area = 0 
        for (hsv_lower, hsv_upper, color) in COLORS:
            contours = rc_utils.find_contours(image, hsv_lower, hsv_upper)
            largest_contour = rc_utils.get_largest_contour(contours)
            if largest_contour is not None:
                contour_area = rc_utils.get_contour_area(largest_contour)
                if contour_area > color_area:
                    color_area = contour_area
                    color_name = color

        return color_name, color_area
    

def detect_AR_Tag(image):
    markers = []

    corners, ids, _ = detector.detectMarkers(image)
    
    for x in range(len(corners)):
        current_corners = corners[x][0]

        orientation = ""
        if current_corners[0][0] == current_corners[1][0]: 
            if current_corners[0][1] > current_corners[1][1]: 
                orientation = "LEFT"
            else: 
                orientation = "RIGHT"
        else: 
            if current_corners[0][0] > current_corners[1][0]:
                orientation = "DOWN"
            else: 
                orientation = "UP"
            
        area = abs((current_corners[2][0] - current_corners[0][0])) * abs((current_corners[2][1] - current_corners[0][1]))

        current_marker = ARMarker(ids[x][0], current_corners, orientation, area)
        markers.append(current_marker)


    return markers, image

def right():
    global speed, angle, count
    count += 1
    speed = 0.6
    angle = 0.7
    
def left():
    global speed, angle, count
    count += 1
    speed = 0.6
    angle = -0.7

def wall_follower():
    global speed, max_speed, angle
    global setpoint, w, x0, x1
    #[insert old wall_follower code]
  
    # Initialize variables
    speed = 0
    angle = 0
    max_speed = 0.09
    
    scan = rc.lidar.get_samples()
    left = rc_utils.get_lidar_average_distance(scan, 315, 30)
    right = rc_utils.get_lidar_average_distance(scan, 45, 30)

    Kp = -0.0015 #-0.00444 #-0.00443 #-0.0044 #-0.00446  #-0.006 #-0.009 #-0.013 
    error = left - right

    angle = Kp * error
    angle = rc_utils.clamp(angle, -1, 1)
    return angle
    #rc.drive.set_speed_angle(0.72, angle)

    # Set initial driving speed and angle
    #rc.drive.set_speed_angle(speed, angle)
def start():
    global speed, angle, count
    speed = 0
    angle = 0
    count = 0
    rc.set_update_slow_time(0.1)
    print("in")
    
def update():
    global speed, angle, count
    global marker
    global markers, count
    image = rc.camera.get_color_image()
    markers = rc_utils.get_ar_markers(image)
    #print(count)
    #print(markers.get_orientation())
    for marker in markers:
        #print("in")
        # print(f"Marker ID: {marker.get_id()}")
        # print(f"Marker Corners: {marker.get_corners_aruco_format()}")
        # print(f"Marker Color: {marker.get_color()}")
        #print(f"Marker Orientation: {marker.get_orientation()}")
        # print("==================================")
        #print(str(marker.get_orientation()))
        if str(marker.get_orientation()) == "Orientation.RIGHT" and count <= 30:
            right()
                #left()
                #print("finih right")
            # print("going right")
        elif str(marker.get_orientation()) == "Orientation.LEFT" and count <= 30:
            left()
        else:
            if count > 0:
                angle *= -1
            
        #else:
            #count = 0
            #angle = wall_follower()
            #print("following wall")
    print(angle)
    rc.drive.set_speed_angle(0.6, angle)
    
def update_slow():
    pass
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
