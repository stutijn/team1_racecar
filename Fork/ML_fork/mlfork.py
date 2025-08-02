"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [racecar follower] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################
import cv2
import os
import time


from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils
########################################################################################
# Global variables
########################################################################################
import racecar_utils as rc_utils
rc = racecar_core.create_racecar()

# Declare any global variables here
default_path = 'arrows' # location of model weights and labels
model_name = 'arrowobstacles.tflite'
label_name = 'objects.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.8
NUM_CLASSES = 2

#variables for following
setpoint = 0
w = 320

STATES = ["right", "left", "waiting"]


########################################################################################
# Functions
########################################################################################
def start():
    global w, x0, x1
    global frame, image, objs, labels, interpreter, count, current_state, previous_state
    # STEP 1: Load model and labels using pycoral.utils
    print('Loading {} with {} labels.'.format(model_path, label_path))
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = read_label_file(label_path)
    global inference_size
    count = 0
    current_state = "waiting"
    previous_state = "waiting"
    #print(labels)
    
    inference_size = input_size(interpreter)

def update_slow():
    pass

# Main function
def update():
    global w, x0, x1
    global frame, image, objs, labels, interpreter
    global inference_size
    
    inference_size = input_size(interpreter)

    # STEP 2: Open webcam
    frame = rc.camera.get_color_image()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, inference_size)
    

    # STEP 5: Let the model do the work
    #ime_cp2 = time.time()
    run_inference(interpreter, rgb_image.tobytes())
    #time_cp3 = time.time()
    
    # STEP 6: Get objects detected from the model
    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]
    #print(f"obj1: {type(objs)}")
    # STEP 7: Label detected objects to frame
    image, x0, x1, label = append_objs_to_img(frame, inference_size, objs, labels)

    # STEP 8: Show labeled image to screen
    cv2.imshow('frame', image)

    wall_follower()
    sign_detect()

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


def sign_detect():
    global speed, max_speed, angle, prev_angle
    global w, x0, x1
    global count
    global previous_state, current_state
    
    
    if frame is not None:
        max_score = 0.65
        actual_obj = None
        label = None
        for obj in objs:
            if obj.score > max_score:
                max_score = obj.score
                actual_obj = obj
                _, x0, x1, label = append_objs_to_img(frame, inference_size, objs, labels)
                #print(f"obj2: {type(obj)}")
            else:
                print("HAHAHAH u failed")
        
        if actual_obj is not None:     
            print(label)
            if label == "Left":
                speed, angle = Left()
            elif label == "right":
                speed, angle = right()
        else:
            angle = wall_follower()
            print("no label detected")
    #     if previous_state == "left" or previous_state == "right":
    #         count += 1
    #     if count>= 11 and previous_state == "left":
    #         count = 0
    #         angle = 1
    #         previous_state = "waiting"
    #     elif count >= 10 and previous_state == "right":
    #         count = 0
    #         angle = -1
    #         previous_state = "waiting"
    # print(count)
        # if previous_state = "left" and count > 
    print(angle)
    rc.drive.set_speed_angle(0.72, angle)


            
# [FUNCTION] Modify image to label objs and score
def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    x0 = None
    x1 = None
    label = ""
    for obj in objs:
        #print(f"obj: {type(obj)}")
        if obj.score > 0.6: # only draw item if confidence > 75%
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label = '{}'.format(labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        else:
            x0 = None
            x1 = None
            label = None
    print(label)    
    return cv2_im, x0, x1, label

def Left():
    global previous_state, current_state
    speed = 0.7
    angle = -1
    # current_state = "left"
    # previous_state = "left"
    
    return speed,angle 
def right():
    global previous_state, current_state
    speed = 0.7
    angle = 1
    # current_state = "right"
    # previous_state = "right"
    return speed,angle 
    
if __name__ == '__main__':
    rc.set_start_update(start, update, update_slow)
    rc.go()
