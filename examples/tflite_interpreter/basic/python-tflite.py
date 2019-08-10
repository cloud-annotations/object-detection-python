# Matthew Dunlop, August 2018
# https://github.com/mdunlop2
#
# Contact:
# https://www.linkedin.com/in/mdunlop2/

import glob
import os

import argparse

# ~~ tmp
import importlib
importlib.reload(vis_util)
importlib.reload(models)
# ~~ tmp
from examples.tflite_interpreter.basic.utils import visualization_utils as vis_util
from examples.tflite_interpreter.basic.utils import cacli_models as models

# Directory in which this example takes place
EXAMPLE_DIR = "examples/tflite_interpreter/basic/"

# Optional User Inputs
# model directory, straight from !cacli download
MODEL_DIR = EXAMPLE_DIR + "model/sample_model/"
MINIMUM_CONFIDENCE = 0.01
PATH_TO_TEST_IMAGES_DIR = EXAMPLE_DIR + "model/test_images"
PATH_TO_OUTPUT_DIR = EXAMPLE_DIR + "model/output"

TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))


MODEL_PATH = MODEL_DIR + "model.tflite"
MODEL_ANCHOR_PATH = MODEL_DIR + "anchors.json"
MODEL_LABEL_PATH = MODEL_DIR + "labels.json"

# Load model and allocate tensors
model_interpreter = models.initiate_tflite_model(MODEL_PATH)
# Load mobilenet-v1 anchor points
anchor_points = models.json_to_numpy(MODEL_ANCHOR_PATH)
# Load Category Index
label_list = models.json_to_numpy(MODEL_LABEL_PATH)

CATEGORY_INDEX = { i : {"name" : label_list[i]} for i in list(range(len(label_list))) }

for image_path in TEST_IMAGE_PATHS:
  models.detect_objects(model_interpreter,
                        image_path,
                        CATEGORY_INDEX,
                        anchor_points,
                        MINIMUM_CONFIDENCE,
                        save_dir=PATH_TO_OUTPUT_DIR)