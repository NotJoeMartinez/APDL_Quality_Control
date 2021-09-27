
import os, sys, re, subprocess, argparse
import numpy as np
import pandas as pd 
from PIL import Image, ImageOps
from tensorflow import keras

# progress bar
def progress(count, total, status=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
  sys.stdout.flush()

"""
Itterates through all test images, prints predictions, confidence levels 
and wheather it actually  predicted it correctly it should return a pandas dataframe
 """
def test_all_imgs(model, class_names, test_data_path, size):

  pandas_data = []
  # makes list of test data paths
  data_paths = []
  for root, dirs, files in os.walk(test_data_path):
    for name in files:
      data_paths.append(os.path.join(root, name))

  # Makes preditctions of every image in the data paths list
  print("Running Tests")
  for count, img_path in enumerate(data_paths):

    progress(count,len(data_paths))

    # make_activation_map(model,img_path, class_names) # Makes activation maps
    temp_data = []
    data = np.ndarray(shape=(1, size[0], size[1], 3), dtype=np.float32)
    image = Image.open(img_path)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    normalized_image_array = image_array.astype(np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    label_prediction = np.argmax(prediction[0])

    test_file_name = re.findall("\/.*\/(.*\.jpg)", img_path)
    parent_dir = re.findall("\/.*\/(.*)\/.*\/*.jpg", img_path)

    # find the parent directory applying it to the rubric then get the index of the class name
    prediction_truth_index = class_names.index(parent_dir[0])
    prediction_truth = class_names[prediction_truth_index]

    if class_names[label_prediction] == prediction_truth:

      temp_data.append("True")
    else:
      temp_data.append("False")

    temp_data.append(class_names[label_prediction]) 
    temp_data.append(prediction_truth) 
    # temp_data.append(100 * np.max(prediction[0])) 
    temp_data.append(np.max(prediction[0])) 
    temp_data.append(f"{parent_dir[0]}/{test_file_name[0]}")

    pandas_data.append(temp_data)
  return pandas_data 

