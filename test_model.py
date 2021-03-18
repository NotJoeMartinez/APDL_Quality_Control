import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import datetime as dt

import os, random



model = keras.models.load_model("models/03_18_1:03PM")
model.summary()

# Testing begins here
img_height = 180
img_width = 180
class_names = ['broken_wire', 'not_broken_wire']

fpath = "datasets/testing/test_broken_wire/"
test_images = [fpath+"broken_wire.png"]

for i in range(0,9):
  test_img_path = fpath + random.choice(os.listdir(fpath))
  test_images.append(test_img_path)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Builds the plot with images of random images and one image of a broken wire

plt.figure(figsize=(15, 15))
for count, x in enumerate(test_images):
  img = keras.preprocessing.image.load_img(
      x, target_size=(img_height, img_width)
  )

  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  # plot building 
  ax = plt.subplot(5, 3, count + 1)
  img = mpimg.imread(x)
  plt.imshow(img)
  plt.title(class_names[np.argmax(score)] + " {:.2f}".format(100 * np.max(score)))
  plt.axis("off")

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )


plt.show()