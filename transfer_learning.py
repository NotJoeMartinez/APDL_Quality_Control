import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import pathlib
import datetime as dt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger




model_name = "mobilenet_v3_small_100_224"

model_handle_map = {
"inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/classification/5",
	"inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
	"mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
	"mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
	"mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
	"mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
	"mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
	"mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
	"mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

model_image_size_map = {
	"inception_v3": 299,
	"inception_resnet_v2": 299,
	"nasnet_large": 331,
	"pnasnet_large": 331,
}

model_handle = model_handle_map.get(model_name)
pixels = model_image_size_map.get(model_name, 224)

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 32 

## Set up the dataset
data_dir = "datasets/training" 
data_dir = pathlib.Path(data_dir)

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
									 interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		**datagen_kwargs)

valid_generator = valid_datagen.flow_from_directory(
		data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

train_datagen = valid_datagen

train_generator = train_datagen.flow_from_directory(
		data_dir, subset="training", shuffle=True, **dataflow_kwargs)

## Defining the model
do_fine_tuning = True 
print("Building model with", model_handle)
model = tf.keras.Sequential([
		# Explicitly define the input shape so the model can be properly
		tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
		hub.KerasLayer(model_handle, trainable=do_fine_tuning),
		tf.keras.layers.Dropout(rate=0.2),
		tf.keras.layers.Dense(train_generator.num_classes,
		kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

## Training the model
model.compile(
	optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
	loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
	metrics=['accuracy'])


steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit(
		train_generator,
		epochs=50, steps_per_epoch=steps_per_epoch,
		validation_data=valid_generator,
		validation_steps=validation_steps).history

now = dt.datetime.now()
model_dir="models/{}/".format(now.strftime("%m_%d_%-I:%M:%S%p"))
checkpoint_name= 'CNNv2.e{epoch:03d}.val_acc_{val_accuracy:01.5f}.h5' 


if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
filepath = os.path.join(model_dir, checkpoint_name)



checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy",
                        verbose=1, save_best_only=True, mode="max")

early = EarlyStopping(monitor="val_loss",
                      mode="min", patience=12)
model.save(model_dir)

print("Model for {} training set saved in {}".format(data_dir,model_dir))