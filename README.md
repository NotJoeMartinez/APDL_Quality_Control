# APDL_Quality_Control
This project is intended to help with the quality control process of manufacturing calorimeters. 

### Installation 
It's important to specify this version of python as Tenserflow has not been working on the latest release 
```bash
git clone https://github.com/NotJoeMartinez/APDL_Quality_Control
cd APDL_Quality_Control 
virtualenv env --python=python3.9.2
source env/bin/activate 
pip install -r requirements.txt
```

If you still get an error installing tenserflow deactivate and remove
```bash
pip install --upgrade tensorflow
```


### Using the existing dataset
To use the classified dataset found `scripts/original.tar.gz`  use the linux `tar` command.
```bash
cp scripts/original.tar.gz  datasets/
tar -zxvf datasets/original.tar.gz
rm datasets/original.tar.gz
```

### `augment_imgs.py`
Before using this to augment images you must first make sure that you have the original dataset in the 
datasets/training/ directory. This script requires no arguments and once run will split the training 
dataset from the testing dataset then augment every image in each subdirectory of `training` to 200
images. 

### `train_model.py`
This file is what trains our model and saves it to the `models/` directory. It requires no
arguments and produces a matplotlib graph showing the training/validation accuracy which can 
be turned off by removing `show_model_details()` from the last line

### `train_mobilenet.py`
This file does the same thing as `train_model.py` except it uses a pre-trained mobile net model
which is what Teachable Machine uses. This method of training has been the most successful but it's 
a black box as far as how it produces it's results. 

### `test_model.py` 
This file takes no arguments but has several functions that can be commented in/out depending on what type of tests you want to do.

By default it looks for the most recently trained model in the `models/` directory.
```python 
    model = keras.models.load_model(f"models/{find_most_recent('models')}")
```
But this can be changed by hard coding the path to the model you want to test.
```python
    model = keras.models.load_model("path/to/model")
```
- `test_all_imgs(model, class_names, test_data_path):`
This function will produce a confusion matrix of the models performance on a specified directory. 
The specified directory which should be passed in as a string. The `class_names` argument 
should be passed in as an array of labels. 

- `random_test_plot(model, class_names, test_data_path)`
This one works similar to `test_all_imgs()`  except it does not plot a confusion matrix or check it's work 
against a rubric. It simply picks 6 random images and produces a prediction/confidence score on a plot. 
This is great for quickly testing the models performance through visual intuition. 


### `models/`
This is where models are saved after training and retrieved for testing.
`train_model.py` models will save the Teserflow model in a directory with the
format `MM_DD_H:MM:SSAM/PM`. 

Ex:
```text
03_24_6:45:34PM
├── assets
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```


### `datasets/` 
This directory is split up into a `testing/`, `training/` & `original/` sub directory
each has a child directory that corresponds to it's name. 
- `original/` is for manually classified images an should never be modified 
- `training/` is contains augmented & original images, each sub directory should have over a hundred images
- `testing/` contains images separated from training before augmentation and each sub directory should have at least two image


### `scripts/separate_datasets.py`
Separates training images into a testing directory while preserving 
parent their respective parent directories, this is done so that the model is not
tested on images that it is trained on. This script is not meant to be called directly
As it's easy to forget that you've called it and it "destructively" modifies the
training directory; Rather it's called by `augment_images.py` before augmenting images. 






