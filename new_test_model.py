import os, sys, random, pathlib, re
from os import path
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
import datetime as dt
from PIL import Image, ImageOps
from scripts.find_most_recent import find_most_recent


subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)
test_data_path = "datasets/testing/"
class_names = ["AllWires", "BrokenWires", "FooBar", "Glue", "NoWires", "OneThirdsWires", "TwoThirdsWires"]
now = dt.datetime.now().strftime("%m_%d_%I:%M:%S%p")


def main(class_names=class_names, test_data_path=test_data_path, size=(480,480)):
	model_path = sys.argv[1]
	model = keras.models.load_model(f"{model_path}") 
	tested_images = test_all_imgs(model, class_names, test_data_path, size) 



def test_all_imgs(model, class_names, test_data_path, size):
	data_paths = []
	for root, dirs, files in os.walk(test_data_path):
		for name in files:
			data_paths.append(os.path.join(root, name))
	
	for image_path in data_paths:
		parent_dir = re.findall("\/.*\/(.*)\/.*\/*.jpg", image_path)
		data = np.ndarray(shape=(1, size[0], size[1], 3), dtype=np.float32)
		image = Image.open(image_path)
		image = ImageOps.fit(image, size, Image.ANTIALIAS)
		image_array = np.asarray(image)
		normalized_image_array = image_array.astype(np.float32)
		data[0] = normalized_image_array
		prediction = model.predict(data)
		label_prediction = class_names[np.argmax(prediction[0])]
		print(prediction,label_prediction, parent_dir)



if __name__ == '__main__':
	main()