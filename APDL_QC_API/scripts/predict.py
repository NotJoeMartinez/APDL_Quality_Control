import os, sys, random, pathlib, re
import PIL
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import subprocess
import pandas as pd 
import datetime as dt
from PIL import Image, ImageOps
import logging, sys



def main():

    # see if logging was enabled
    if sys.argv[1] == "log":
        log_stuff()

    model = keras.models.load_model(f"model/05_20_7:49:27PM")
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    class_names = ['Broken Wire', 'Glue', 'Good', 'No Wires', 'One Third Wire', 'Two Third Wires', 'Unknown Debris']


def make_predictions(probability_model, class_names, image_path):
    # declairs an empty ndarray 
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # converts image from PIL.Image.Image to numpy.ndarray
    size = (224, 224)
    image = ImageOps.fit(Image.open(image_path), size, Image.ANTIALIAS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array


    predictions = probability_model.predict(data)

    label_prediction = class_names[np.argmax(predictions[0])]

    prediction_confidence = 100*np.max(predictions[0])


def log_stuff():
    now = dt.datetime.now().strftime("%m_%d_%-I:%M:%S%p")
    logging.basicConfig(filename=f'logs/{now}_predict.log', encoding='utf-8', level=logging.DEBUG)


if __name__ == '__main__':
    main()