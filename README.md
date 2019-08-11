# Object Detection Python Examples

Run your models trained using [Cloud Annotations](https://github.com/cloud-annotations/training) with python.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites and installing
Install the required packages in `requirement.txt`

Creating a new virtual environement is recommended.

```
conda create -n object_detection python=3.7
conda activate object_detection
```

Git clone the repo and change directory into it. Then pip install the packages in `requirement.txt`.
```
cd directory/you/want/to/clone/into
git clone https://github.com/cloud-annotations/object-detection-python.git
cd object-detection-python
pip install -r requirement.txt
```

### Test if everything is working
I have supplied a test model and some test images. This should output the images with detection boxes and labels in jpg format in 'examples/tflite_interpreter/basic/model/output'

```
cd examples/tflite_interpreter/basic/
python python-tflite.py
```

## Tflite Object Detection
Currently, `python-tflite.py` supports using Mobilenet-V1 SSD models trained using Cloud Annotations.

Note: to find a list of all models trained do:
```
cacli list
```

To use a custom model, perform
```
cacli download <model_name>
```
For example, if the downloaded files were saved to `/path/to/<model_name>` :
* Our tflite model is stored in `<model_name>/model_android/model.tflite`
* Our tflite anchors file is stored in `<model_name>/model_android/anchors.json`
* Our tflite labels file is stored in `<model_name>/model_android/labels.json`

Change directory to the root of this git.
```
cd examples/tflite_interpreter/basic/
python python-tflite.py --MODEL_DIR /path/to/<model_name>/model_android
```
This script calls the tflite model interpreter for inference on all .jpg files inside the directory `PATH_TO_TEST_IMAGES_DIR`.

Similary the output .jpg files are storesd in `PATH_TO_OUTPUT_DIR`.

We can also specify the minimum confidence (score) for a given detection box to be displayed with `MINIMUM_CONFIDENCE`.

Finally:
```
python python-tflite.py \
--MODEL_DIR /path/to/<model_name>/model_android \
--PATH_TO_TEST_IMAGES_DIR /path/to/test/images \
--PATH_TO_OUTPUT_DIR /path/to/output/images \
--MINIMUM_CONFIDENCE 0.01

```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


