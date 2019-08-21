# Matthew Dunlop, August 2018
# https://github.com/mdunlop2
#
# Contact:
# https://www.linkedin.com/in/mdunlop2/

'''
Set of utility functions to help with loading and testing a mobilenet-v1 model trained with
https://github.com/cloud-annotations/training 
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import math as m
import time as t

import os
import PIL
from PIL import Image

from errno import EEXIST
from pathlib import Path

import matplotlib.pyplot as plt

import glob

# Utility Packages
# from examples.tflite_interpreter.basic.utils import visualization_utils as vis_util
from utils import visualization_utils as vis_util

def json_to_numpy(path):
    '''
    Opens anchors.json and labels.json
    Returns numpy array.
    '''
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def load_image_into_numpy_array(image, reg = False):
    '''
    Takes PIL image (only supports batch size of 1 currently)
    Optionally regularises to [0,1]
    Returns (1,im_height, im_width, 3) numpy array
    '''
    (im_width, im_height) = image.size
    if reg:
        return np.array(image.getdata()).reshape(
        (1,im_height, im_width, 3)).astype(np.float32)/255
    else:
        return np.array(image.getdata()).reshape(
            (1,im_height, im_width, 3)).astype(np.float32)

def initiate_tflite_model(MODEL_PATH):
    # initiates the tflite model interpreter and allocates
    # tensors. To be performed before inference.
    interpreter = tf.lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def call_tflite_model(interpreter, input_img):
    '''
    Designed to work with Mobilenet-V1 SSD 300 models trained on Cacli

    Returns inference classes, boxes, scores
    
    Args:
    interpreter : (tf.lite.Interpreter) applied to MODEL_PATH
                  Should have run model.allocate_tensors() previously.
                  See initiate_tflite_model.
                  ~~TODO~~
                  Could possibly offload get_input_details to intiate_tflite_model

    input_img   : (PIL image) PIL image

    Returns:
    classes     : array(N,) Raw output classes from the mobilenet-v1 SSD model
                Consists of floats, need to perform rounding.
                ~~TODO~~
                Check if this is in fact quantised, what transform needs to be performed?

    boxes       : array(N, 4) Raw box co ordinate output from quantised mobilenet-v1 SSD model.
                Is quantised, so consists of floats.

    scores      : array(N,) Raw inference score output from quantised mobilenet-v1 SSD model.
                Consists of floats, need to perform rounding.
                ~~TODO~~
                Check if this is in fact quantised, what transform needs to be performed?

    '''
    img_column, img_row = 300, 300
    input_img = input_img.resize((img_row, img_column))
    # Convert to regularised numpy (values in [0,1])
    x_matrix = load_image_into_numpy_array(input_img, reg = True)
    # get interpreter details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # convert image to input format
    input_data = x_matrix.reshape(input_details[0]['shape'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))[:,0]
    scores = np.squeeze(interpreter.get_tensor(output_details[1]['index']))[:,1]
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    return classes, boxes, scores



def detect_objects(model, IMAGE_PATH, CATEGORY_INDEX, ANCHOR_POINTS, MINIMUM_CONFIDENCE, SAVE_DIR = None):
    '''
    Adapted from tensorflow slim.
    Performs object detection inference on given image (.jpg in IMAGE_PATH)
    with a tflite Mobilenet-V1 model which has already been allocated tensors

    Args:
    model              : (tf.lite.Interpreter) applied to MODEL_PATH
                        Should have run model.allocate_tensors() previously.
                        See initiate_tflite_model

    IMAGE_PATH         : (String) Path to image (.jpg) on which to perform inference

    CATEGORY_INDEX     : (Dictionary) Dictionary of dictionaries, key: position in labels.json
                        Each sub-dictionary has "name" field which will be displayed
                        beside the bounding box

    ANCHOR_POINTS      : array(N, 4) Shaped numpy array representing each of the
                        anchor points provided for Mobilenet-v1 SSD in anchors.json

    MINIMUM_CONFIDENCE : (Float) Minimum score permissible for box to be considered for
                        display. Mobilenet-V1 SSD tends to provide quite low probabilities.

    SAVE_DIR           : (String) (Optional) Directory to save output inference images to.
                        If None, will print the plot to CLI with Matplotlib
    '''


    image = Image.open(IMAGE_PATH)
    start = t.time()
    classes, boxes, scores = call_tflite_model(model, image)
    print("Inference time: {}".format(t.time()-start))
    image_np = load_image_into_numpy_array(image)
    image_np_reg = load_image_into_numpy_array(image, reg=True)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Convert the quantised boxes to normalised
    ty = boxes[:,0] / float(10)
    tx = boxes[:,1] / float(10)
    th = boxes[:,2] / float(5)
    tw = boxes[:,3] / float(5)

    yACtr = ANCHOR_POINTS[:,0]
    xACtr = ANCHOR_POINTS[:,1]
    ha    = ANCHOR_POINTS[:,2]
    wa    = ANCHOR_POINTS[:,3]

    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    yCtr = ty * ha + yACtr
    xCtr = tx * wa + xACtr

    yMin = yCtr - h / float(2)
    xMin = xCtr - w / float(2)
    yMax = yCtr + h / float(2)
    xMax = xCtr + w / float(2)
    
    boxes_normalised = [yMin, xMin, yMax, xMax]
    print("-"*10)
    print("Inference Summary:")
    print("Highest Score: {}".format(np.max(scores) ) )
    print("Highest Scoring Box: {}".format(np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)]) )
    print("-"*10)
    print("Image shape: {}".format(np.squeeze(image_np).shape))
    print("Boxes shape: {}".format(np.transpose(np.squeeze(boxes_normalised)).shape))
    print("Classes shape: {}".format(np.round(np.squeeze(classes)).astype(np.int32).shape ))
    print("Scores shape: {}".format(np.squeeze(scores).shape))
    fig = plt.figure()
    out_image = vis_util.visualize_boxes_and_labels_on_image_array(
        np.squeeze(image_np),
        np.transpose(np.squeeze(boxes_normalised)),
        np.round(np.squeeze(classes)).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8,
        ret = True)
    
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(out_image/255)
    if SAVE_DIR:
      output_dir = str(SAVE_DIR+'/{}'.format(IMAGE_PATH))
      mkdir_p(Path(output_dir).parent)
      plt.savefig(output_dir, dpi = 62)
      plt.close(fig)
      print("Image Saved")
      print("="*10)

def mkdir_p(mypath):
  '''Creates a directory. equivalent to using mkdir -p on the command line'''
  try:
      os.makedirs(mypath)
  except OSError as exc: # Python >2.5
      if exc.errno == EEXIST and os.path.isdir(mypath):
          pass
      else: raise

print("Util imports Successful")

