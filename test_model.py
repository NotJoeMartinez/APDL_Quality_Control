import os, random, sqlite3, pathlib, re
import PIL

import numpy as np
import datetime as dt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import subprocess

import pandas as pd 

# load model
model = keras.models.load_model("models/03_22_3:36PM")
# model.summary()

img_height = 180
img_width = 180
class_names = ['broken wire', 'glue', 'good', 'missing wire', 'unknown debris']
test_data_paths = "datasets/testing/test_all_features"
rubric_dict =  {
  "broken_wire": "broken wire",
  "glue": "glue",
  "good": "good",
  "missing_wire": "missing wire",
  "unknown_debris": "unknown debris"
}

# finds image names out of the training dataset then returns their parent directory 
def find_image_label(img_name, file_path="datasets/training/all_features", ):
  foo = subprocess.check_output("find {} -name {}".format(file_path,img_name) , shell=True)
  parent_dir = re.findall("\/(\w*)\/", str(foo))
  
  try:
      return "{}".format(parent_dir[1])
  except IndexError:
    pass

"""
Itterates through all test images, prints predictions, confidence levels 
and wheather it actually  predicted it correctly it should return a pandas dataframe
"""
def test_all_imgs():

  pandas_data = []
  # makes list of test data paths
  data_paths = []
  for root, dirs, files in os.walk(test_data_paths):
    for name in files:
      data_paths.append(os.path.join(root, name))
  
  # Makes preditctions of every image in the data paths list
  for count, test_img in enumerate(data_paths):
    temp_data = []

    # turn test_img into a preprocessed keras object or something
    img = keras.preprocessing.image.load_img(
        test_img, target_size=(img_height, img_width)
    )

    # turn image to image array an made predicitons using the model
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # declare some more varabables 
    test_file_name = re.findall("\/.*\/(.*\.png)",test_img)
    model_prediction = class_names[np.argmax(score)]

    # check stuff against reality 
    try:
      true_file_label = find_image_label(test_file_name[0])
      if rubric_dict[true_file_label] == model_prediction:
        prediction_truth = "True" 
      else:
        prediction_truth = "False" 
      
      # print(
      #   "Prediction: {}, Truth: {}, Percent Confidence: {:.2f}".format(model_prediction, 
      #    prediction_truth, 
      #   100 * np.max(score))
      # )

      temp_data.append(model_prediction) 
      temp_data.append(prediction_truth) 
      temp_data.append(100 * np.max(score)) 

      pandas_data.append(temp_data)

    except (IndexError, KeyError):
      pass
  return pandas_data 

yeet = test_all_imgs()
df = pd.DataFrame(yeet, columns = ['prediction','prediction_truth','confidence'])
print(df)


# Builds the plot with images of random images and one image of a broken wire
def random_test_plot():
  test_images = []
  for i in range(0,9):
    test_img_path = fpath + "/" + random.choice(os.listdir(fpath))
    test_images.append(test_img_path)

  plt.figure(figsize=(15, 15))
  for count, x in enumerate(test_images):
    img = keras.preprocessing.image.load_img(
        x, target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # plot building 
    ax = plt.subplot(5, 3, count + 1)
    img = mpimg.imread(x)
    plt.imshow(img)
    plt.title(class_names[np.argmax(score)] + " {:.2f}".format(100 * np.max(score)))
    plt.axis("off")

  plt.show()


