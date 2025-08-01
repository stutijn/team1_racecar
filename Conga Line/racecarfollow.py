"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: racecarfollow.py << [Modify with your own file name!]

Title: Conga Line: Racecar follower

Author: TEAM 1

Purpose: Using ML model, be able to identify a car ahead and follow it accordingly. 

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

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils
########################################################################################
# Global variables
########################################################################################
import racecar_utils as rc_utils
rc = racecar_core.create_racecar()

# Declare any global variables here
default_path = 'models' # location of model weights and labels
model_name = 'model.tflite'
label_name = 'labels.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.8
NUM_CLASSES = 1

#variables for following
setpoint = 0
w = 640
interpreter = None
prev_angle = 0
########################################################################################
# Functions
########################################################################################

def start():
    global w, x0, x1
    global frame, image, objs, labels, interpreter
    # STEP 1: Load model and labels using pycoral.utils
    print('Loading {} with {} labels.'.format(model_path, label_path))
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = read_label_file(label_path)
    global inference_size
    
    inference_size = input_size(interpreter)

def update_slow():
    pass

# Main function
def update():
    global w, x0, x1, interpreter
    global frame, image, objs, labels
    global inference_size
    
    inference_size = input_size(interpreter)

    # STEP 2: Open webcam
    frame = rc.camera.get_color_image()

    # STEP 3: Loop through webcam camera stream and run model
    #not necessary 
        
    # STEP 4: Preprocess image to the size and shape accepted by model
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, inference_size)
    
    run_inference(interpreter, rgb_image.tobytes())

    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]

    image, x0, x1 = append_objs_to_img(frame, inference_size, objs, labels)
    
    rc.display.show_color_image(frame)

    car_follower()

def car_follower():
    global speed, max_speed, angle, prev_angle
    global setpoint, w, x0, x1

    # Initialize variables
    speed = 0
    angle = 0
    max_speed = 0.09

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    if frame is not None:
        max_score = 0.8
        actual_obj = None
        for obj in objs:
            if obj.score > max_score:
                max_score = obj.score
                actual_obj = obj
                _, x0, x1 = append_objs_to_img(frame, inference_size, objs, labels)
                
            else:
                print("HAHAHAH u failed")
        
       #: {type(present_value)}; Setpoint type: {type(setpoint)}")
        if actual_obj is not None:     
            print("car found")
            setpoint = w // 2
            if x0 is not None and x1 is not None:
                present_value = (x0 + x1) // 2
                error = setpoint - present_value
                kp = -0.004
                prev_angle = angle
                angle = kp*error
                angle = rc_utils.clamp(angle ,-1,1)
            else:
                present_value = None
                
            
        else:
            if prev_angle < 0:
                angle = -1
            else:
                angle = 1
    print(angle)
    rc.drive.set_speed_angle(0.8, angle)

# [FUNCTION] Modify image to label objs and score
def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    x0 = None
    x1 = None
    for obj in objs:
        #print(f"obj: {type(obj)}")
        if obj.score > 0.6: # only draw item if confidence > 75%
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        else:
            x0 = None
            x1 = None
    
    return cv2_im, x0, x1
    

if __name__ == '__main__':
    rc.set_start_update(start, update, update_slow)
    rc.go()
