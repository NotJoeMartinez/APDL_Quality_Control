import os, sys, random, pathlib, re
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

def main():
  subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)
  # model = keras.models.load_model(f"models/{find_most_recent('models')}")
  model = keras.models.load_model("models/converted_keras/keras_model.h5")
  test_data_path = "datasets/testing/jpg"
  model.summary()
  class_names = ['Broken Wire', 'Glue', 'Good', 'No Wires', 'One Third Wire', 'Two Third Wires', 'Unknown Debris']
  # random_test_plot(model, class_names, test_data_path)
  tested_images = test_all_imgs(model, class_names, test_data_path) 
  df = pd.DataFrame(tested_images, columns = ['predicted','actual','confidence','path'])
  plot_confusion_matrix(df)  


def plot_confusion_matrix(df):
  import seaborn as sn

  print(df)
  confusion_matrix = pd.crosstab(df['predicted'], df['actual'], rownames=['Predicted'], colnames=['Actual'])
  sn.heatmap(confusion_matrix, annot=True)
  # plt.fi
  plt.show()


def find_most_recent(directory):
  now = dt.datetime.now()
  dir_list = os.listdir(directory)
  datetimes = []
  for x in dir_list:
    dir_dt = dt.datetime.strptime(x, '%m_%d_%I:%M:%S%p')
    datetimes.append(dir_dt)

  most_recent = max(dt for dt in datetimes if dt < now)
  return most_recent.strftime("%m_%d_%-I:%M:%S%p")
  # return most_recent

def plot_value_array(predictions_array, class_names):
  plt.grid(False)
  plt.xticks(range(7))
  plt.yticks([])
  thisplot = plt.bar(range(7), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')

def plot_image(predictions_array, class_names, img_array):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img_array, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array)))

def test_all_imgs(model, class_names, test_data_path):
  """
  Itterates through all test images, prints predictions, confidence levels 
  and wheather it actually  predicted it correctly it should return a pandas dataframe
  """
  rubric = {
  'broken_wire': 'Broken Wire', 
  'glue': 'Glue', 
  'good': 'Good', 
  'no_wires': 'No Wires', 
  'one_thirds_wires': 'One Third Wire', 
  'two_thirds_wires': 'Two Third Wires', 
  'unknown_debris': 'Unknown Debris'}
  
  pandas_data = []
  # makes list of test data paths
  data_paths = []
  for root, dirs, files in os.walk(test_data_path):
    for name in files:
      data_paths.append(os.path.join(root, name))

  # Makes preditctions of every image in the data paths list
  
  for count, img_path in enumerate(data_paths):
    temp_data = []
    size = (224, 224)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    label_prediction = np.argmax(prediction[0])
    
    # find test image path parent
    test_file_name = re.findall("\/.*\/(.*\.jpg)", img_path)
    parent_dir = re.findall("\/(\w*)\/", img_path)

    # check stuff against reality 
    # if rubric[parent_dir[1]] == class_names[label_prediction]:
    #   prediction_truth = "True" 
    # else:
    #   prediction_truth = "False" 

    # find the parent directory applying it to the rubric then get the index of the class name
    prediction_truth_index = class_names.index(rubric[parent_dir[1]])
    prediction_truth = class_names[prediction_truth_index]
    temp_data.append(class_names[label_prediction]) 
    temp_data.append(prediction_truth) 
    temp_data.append(100 * np.max(prediction[0])) 
    temp_data.append(f"{parent_dir[1]}/{test_file_name[0]}")

    pandas_data.append(temp_data)
  
  return pandas_data 


# Builds the plot with images of random images and one image of a broken wire
def random_test_plot(model, class_names, test_data_path):
    data_paths = []
    for root, dirs, files in os.walk(test_data_path):
        for name in files:
            data_paths.append(os.path.join(root, name))

    random_test_images = random.choices(data_paths, k=6)
    num_rows = 3
    num_cols = 3
    num_images = num_rows*num_cols
    size = (224, 224)
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i, img_path in enumerate(random_test_images):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(img_path)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(prediction[0], class_names, image_array)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(prediction[0], class_names)
        _ = plt.xticks(range(7), class_names, rotation=90)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    sys.exit(0)