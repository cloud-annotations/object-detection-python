# Matthew Dunlop, August 2018
# https://github.com/mdunlop2
#
# Contact:
# https://www.linkedin.com/in/mdunlop2/

import glob
import os

# ~~ tmp
import importlib
importlib.reload(vis_util)
importlib.reload(models)
# ~~ tmp
from examples.tflite_interpreter.basic.utils import visualization_utils as vis_util
from examples.tflite_interpreter.basic.utils import cacli_models as models

# Directory in which this example takes place
EXAMPLE_DIR = "examples/tflite_interpreter/basic/"

MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.01
PATH_TO_TEST_IMAGES_DIR = EXAMPLE_DIR + "model/test_images"
PATH_TO_OUTPUT_DIR = EXAMPLE_DIR + "model/output"

TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

# model directory, straight from !cacli download
MODEL_DIR = EXAMPLE_DIR + "model/sample_model/"
MODEL_PATH = MODEL_DIR + "model.tflite"
MODEL_ANCHOR_PATH = MODEL_DIR + "anchors.json"

CATEGORY_INDEX = {1: {"name" : "N/A", "id" : "1"},\
                  0: {"name" : "plate", "id" : "0"}
                    }

model_interpreter = models.initiate_tflite_model(MODEL_PATH)
anchor_points = models.load_anchors(MODEL_ANCHOR_PATH)




for image_path in TEST_IMAGE_PATHS:
  models.detect_objects(model_interpreter,
                        image_path,
                        CATEGORY_INDEX,
                        anchor_points,
                        MINIMUM_CONFIDENCE,
                        save_dir=PATH_TO_OUTPUT_DIR)

# quick_paths = ["examples/tflite_interpreter/basic/model/test_images/train_img.jpg"]
# for image_path in quick_paths:
#   models.detect_objects(model_interpreter,
#                         image_path,
#                         CATEGORY_INDEX,
#                         anchor_points,
#                         MINIMUM_CONFIDENCE,
#                         save_dir=PATH_TO_OUTPUT_DIR)
