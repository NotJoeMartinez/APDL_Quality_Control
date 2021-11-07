from APDL_QC_API import predictions
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


class_names = ['Broken Wire', 'Glue', 'Good', 'No Wires', 'One Third Wire', 'Two Third Wires', 'Unknown Debris']
model = keras.models.load_model(f"APDL_QC_API/predictions/model/jordan_net")
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

def make_predictions(image_data, class_names=class_names, probability_model=probability_model):
    """Returns a dictionary of tenserflow model predictions"""
    data = np.ndarray(shape=(1, 480, 480, 3), dtype=np.float32)
    # converts image from PIL.Image.Image to numpy.ndarray
    size = (480, 480)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    normalized_image_array = image_array.astype(np.float32)
    data[0] = normalized_image_array
    predictions = probability_model.predict(data)

    all_predictions_dict = get_all_predictions_dict(predictions)
 
    predictions_dict = {
        "Predicted Label": '',
        "Prediction Confidence": '',
        "All Predictions": '',
    } 


    predicticted_label = class_names[np.argmax(predictions[0])]
    prediction_confidence = 100*np.max(predictions[0])

    predictions_dict['Predicted Label'] = predicticted_label 
    predictions_dict['Prediction Confidence']= "{:2.0f}%".format(prediction_confidence) 
    predictions_dict['All Predictions'] = all_predictions_dict 

    return predictions_dict 



def get_all_predictions_dict(predictions_arr, class_names=class_names):   
    """ Makes a dictionary of all the models predictions 

        Format:
        {
            "Prediction label": "[Precentage confidence of model]%"
        }
    """
    
    all_predictions_dict = {}
    for index, pred in enumerate(predictions_arr[0]):
        pred = "{:2.0f}%".format(100*pred)
        all_predictions_dict[class_names[index]] = pred 

    return all_predictions_dict


