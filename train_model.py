import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import datetime as dt

data_dir = input("Dataset Training directory: ")
data_dir = pathlib.Path(data_dir)


batch_size = 32
img_height = 180
img_width = 180


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

# this should be dynamic to the amout of directories there are in data_dir 
num_classes =  5 # sum([len(folder) for r, d, folder in os.walk(data_dir)])
print("This data directory has {} subdirectorys".format(num_classes))

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  # layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.summary()
print("Num Calsses: ",num_classes)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



# Saving model
now = dt.datetime.now()
model_dir="models/{}/".format(now.strftime("%m_%d_%-I:%M%p"))
model.save(model_dir)
print("Model for {} training set saved in {}".format(data_dir,model_dir))



