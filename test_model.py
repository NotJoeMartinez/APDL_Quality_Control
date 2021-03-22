import os, random, sqlite3, pathlib
import PIL

import numpy as np
import datetime as dt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# load model
model = keras.models.load_model("models/03_22_3:36PM")
# model.summary()

img_height = 180
img_width = 180
class_names = ['broken wire', 'glue', 'good', 'missing wire', 'unknown debris']

fpath = input("Testing Path: ")

def test_all_imgs():
  # makes a list of data paths 
  data_paths = []
  for root, dirs, files in os.walk(fpath):
    for name in files:
      data_paths.append(os.path.join(root, name))
    
  # itterates 
  for count, test_img in enumerate(data_paths):
    img = keras.preprocessing.image.load_img(
        test_img, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
        )


test_all_imgs()




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


