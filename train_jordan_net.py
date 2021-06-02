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

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--d1", type="float", dest="d1", default=0.5)
parser.add_option("--d2", type="float", dest="d2", default=0.3)
parser.add_option("--l1_1", type="float", dest="l1_1", default=0.07)
parser.add_option("--l1_2", type="float", dest="l1_2", default=0.007)

(options, args) = parser.parse_args()


#data_dir = "datasets/training" 
data_dir = "datasets/training" 
data_dir = pathlib.Path(data_dir)
#data_dirV = "datasets/validating" 
data_dirV = "datasets/validating" 
data_dirV = pathlib.Path(data_dirV)


batch_size = 128
img_height = 480 
img_width = 480 


# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dirV,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)


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


class_names = train_ds.class_names

print(f'class_names: {class_names}')

# this should be dynamic to the amout of directories there are in data_dir 
num_classes =  7 # sum([len(folder) for r, d, folder in os.walk(mmmmmmmm)])
print("This data directory has {} subdirectorys".format(num_classes))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2","/gpu:3"],
                             cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():

  model = Sequential([
    
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3)),

    layers.Flatten(),

    layers.Dense(128, activation='relu',use_bias=True, kernel_regularizer =tf.keras.regularizers.l1( l=options.l1_1)),
    layers.Dropout(options.d1),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu',use_bias=True, kernel_regularizer =tf.keras.regularizers.l1( l=options.l1_2)),
    layers.Dense(num_classes,activation="softmax")
  ])
  model.summary()
  print("Num Calsses: ",num_classes)

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=100
#epochs=25

import os
#save_dir = f'model_checkpoints/d1_{options.d1}_d2_{options.d2}_l1-1_{options.l1_1}_l1-2_{options.l1_2}/'
#model_name = 'CNNv2.e{epoch:03d}.val_acc_{val_accuracy:01.5f}.h5' 

save_dir = f'model_checkpoints/JD_CNN_v1/'
model_name = 'CNNv2.e{epoch:03d}.val_acc_{val_accuracy:01.5f}.h5' 
print(f"MODEL_NAME HERE LINE 114: {model_name}")

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy",
                        verbose=1, save_best_only=True, mode="max")

early = EarlyStopping(monitor="val_loss",
                      mode="min", patience=12)

csv_logger = CSVLogger('model_loger.csv', append=True, separator=',')
#callbacks_list = [checkpoint, early, csv_logger]
callbacks_list = [checkpoint, csv_logger]


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=callbacks_list
)


# Saving model
now = dt.datetime.now()
model_dir="models/{}/".format(now.strftime("%m_%d_%-I:%M:%S%p"))
model.save(model_dir)
print("Model for {} training set saved in {}".format(data_dir,model_dir))


def show_model_details():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  #plt.subplot(1, 2, 1)
  plt.subplot(2, 1, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  #plt.subplot(1, 2, 2)
  plt.subplot(2, 1, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()
  plt.savefig('performance.png')  


show_model_details()
