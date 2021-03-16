import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import datetime as dt

data_dir = "datasets/training/"
data_dir = pathlib.Path(data_dir)


batch_size = 32
img_height = 180
img_width = 180

print(data_dir, type(data_dir))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  # labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


num_classes = 6

model = keras.models.load_model("models/03_16_5:47PM")
model.summary()

# Testing begins here

test_path = "datasets/training/glue/glue/2AzfCrPr0nI0qDZ66S9uWvdV9OJ4n5pT_2AzfCrPr0nI0qDZ66S9uWvdV9OJ4n5pT_.png"

img = keras.preprocessing.image.load_img(
    test_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(test_path)
imgplot = plt.imshow(img)
plt.show()