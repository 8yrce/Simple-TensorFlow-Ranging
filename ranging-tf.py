
# by 8ryce
"""
Usuage: python3 ranging-tf.py
        Hold an object found on the 'labels.txt' list in front of the camera and follow prompts

  Make sure to have the model provided in the same folder, or specify your model with the model flag
Ranging using tensorflow with a given model. Detection logic based off of the tensorflow detection scripts for stability
super simple ranging example using object area
"""

import numpy as np
"""
#Importing an setting the tensorflow vals for RTX series architecture ( if you dont have an RTX card its fine this wont affect anything )
"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

"""
Opening out cam on cap 0, the built in / first connected cam ( for USB cams check where it is with the 'lsusb' cmd ) 
"""
import cv2
cap = cv2.VideoCapture(0)

"""
Setting up arg parse
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default="ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",help="Path to the model")
args = parser.parse_args()

"""
Import the model into a graph
"""
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  try:
    print("\n\n[INFO] Trying with: {}\n\n".format(args.model))
    with tf.io.gfile.GFile(args.model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

  except Exception as e:
        print("[ERROR]: ").format(e)

# initializing ranging variables
AREA = 0
RANGE_FLAG = False

#Object global vars
OBJECT = 0
PREV_OBJ = -1

"""
ranging
Takes bounding box area to calculate distance
PARAMS: cur_area -current area value for detection

Basically we take a measurement of the object you show at a given point, and then
  save the area, using this we can assume that every time it gets 50% smaller
  its moved twice as far away so long as the object is in the same orientation.
"""
def ranging(cur_area):
  global AREA
  distance = (AREA/cur_area)/2 + 0.5
  print("\n[INFO] Distance: '{0:.2f}' feet\n".format(distance))

"""
detection_handler
Handles all of our detection logic to clean things up in the main loop
PARAMS: boxes -bounding box for top prediction, classes -class of prediction, counter -counter for logic, image_np -image to modify
RETURNS: image_np -updated image, counter -updated counter
"""
def detection_handler(boxes,classes,counter,image_np):
  global AREA
  global OBJECT
  global PREV_OBJ

  top,left,bottom,right = boxes
  #check and see if we have our measurements

  h,w,c = image_np.shape # getting input image sizes
  top = int(top * h)
  left = int(left * w)
  bottom = int(bottom * h)
  right = int(right * w)

  #if we have determined class we want to find
  if classes == OBJECT:
    #Drawing bounding box
    cv2.rectangle(image_np,(left,top),(right,bottom),(0,255,0), 10) # cv2 takes in image, left/top, right/bottom, color, line thickness

    # if we have calculated base area, then range it
    if AREA != 0:
      ranging( (right-left) * (bottom-top) )

  #If we have yet to find the class
  else:
    # all of this works to setup ranging / to avoid people so we can hold the objects
    if counter > -1 and classes != 1:
      if PREV_OBJ == classes:
        counter+=1
      else:
        PREV_OBJ = classes
        counter = 0
      cv2.rectangle(image_np,(75,25),(475,25),(0,0,0),50)
      for i in range(counter):
        cv2.circle(image_np,((75*(i+1)),25),10,((255-(50*i)),255,(255-(50*i))),10)

    if counter > 5:
      print("\nObject selected\n")
      OBJECT = PREV_OBJ
      AREA = (right-left) * (bottom-top)
      counter = -1

  return image_np, counter

"""
Main detection code using our model
"""
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    print("Hold object one foot from the camera for a few seconds.\nThe entire object should be visible in frame")
    counter = 0
    while True:
      # gathering input image from the camera
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection
      try:
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        #Feeding into the detection logic handler
        image_np, counter = detection_handler(boxes[0][0], classes[0][0], counter, image_np)
        
      except Exception as e:
        print("*\n[INFO] Error: {}\n*".format(e))
        exit()
      
      cv2.imshow('Simple ranging v1', image_np)
      cv2.waitKey(1)