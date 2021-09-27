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
now = dt.datetime.now().strftime("%m_%d_%-I%M%S")
print(now)

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--d1", type="float", dest="d1", default=0.5)
parser.add_option("--d2", type="float", dest="d2", default=0.3)
parser.add_option("--l1_1", type="float", dest="l1_1", default=0.07)
parser.add_option("--l1_2", type="float", dest="l1_2", default=0.007)

(options, args) = parser.parse_args()


data_dir = "91321_croped_clean_fdupes/training" 
data_dir = pathlib.Path(data_dir)


batch_size = 128
img_size = 480 


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

class_names = train_ds.class_names

print(f'class_names: {class_names}')

num_classes =  7 

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2","/gpu:3"],
                             cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# no idea why but these need ot be here
d1 = 0.5
d2 = 0.3
l1_1 = 0.07
l1_2 = 0.007

with strategy.scope():

  model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.BatchNormalization(), # added by KL
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
    layers.BatchNormalization(), # added by KL
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3)),

    layers.Flatten(),

    layers.Dense(128, activation='relu',use_bias=True, kernel_regularizer =tf.keras.regularizers.l1( l=l1_1)),
    layers.Dropout(d1),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu',use_bias=True, kernel_regularizer =tf.keras.regularizers.l1( l=l1_2)),
    layers.Dense(num_classes,activation="softmax")
    
])  

  model.summary()
  print("Num Calsses: ",num_classes)

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=100


# Declare model directory
save_dir = f'models/{now}'
model_name = now + 'e{epoch:03d}.val_acc_{val_accuracy:01.5f}.h5' 


# make model checkpoint directory 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Declaring checkpoint 
filepath = "{}/{}".format(save_dir,model_name)

checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy",
                        verbose=1, save_best_only=True, mode="max")

early = EarlyStopping(monitor="val_loss",
                      mode="min", patience=12)

csv_logger = CSVLogger(f'{save_dir}/model_loger.csv', append=True, separator=',')
callbacks_list = [checkpoint, csv_logger]


# running model.fit
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=callbacks_list
)


from testing import plots 
plots.show_model_details(save_dir,history,epochs)


print(save_dir)
from csv import writer
  
# The data assigned to the list 
list_data=[now,model_name,save_dir,epochs,history.history['val_accuracy'],history.history['val_loss'],data_dir]
  
# Pre-requisite - The CSV file should be manually closed before running this code.

# First, open the old CSV file in append mode, hence mentioned as 'a'
# Then, for the CSV file, create a file object
with open('training_hist.csv', 'a', newline='') as f_object:  
    # Pass the CSV  file object to the writer() function
    writer_object = writer(f_object)
    # Result - a writer object
    # Pass the data in the list as an argument into the writerow() function
    writer_object.writerow(list_data)  
    # Close the file object
    f_object.close()
