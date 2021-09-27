import os, sys, random, re, subprocess, argparse
import numpy as np
import tensorflow as tf
import pandas as pd 
from os import path
from PIL import Image, ImageOps
from tensorflow import keras
import plots

def main(args):
  report_name = args.reportname
  model_path = args.modelpath
  size = args.inputsize
  class_names = args.classnames


  try:
    model = keras.models.load_model(f"{model_path}") 
  except IndexError:
    print("You need to specifiy a model")

  # verify overwriting model report
  if path.exists(f"notes/{report_name}.md"):
    overwrite = input("MODEL REPORT EXISTS; Do you want to overwrite? (Y/n): ")
    if overwrite == "n":
      print("Exiting")
      sys.exit() 
    else:
      pass

  # for confusion matrix
  tested_images = test_all_imgs(model, class_names, test_data_path, size) 
  df = pd.DataFrame(tested_images, columns = ['score','predicted','actual','confidence','path'])
  df.to_csv(f'notes/csvs/{report_name}.csv', encoding='utf-8')
  plots.plot_confusion_matrix(df,fig_name=f"notes/imgs/{report_name}.png", show=False)  

  # for random sampleing 
  plots.random_test_plot(model, class_names, test_data_path, report_name, size, show=False)

  # for calulating results
  from scripts.model_reporting import calculate_results
  calculate_results(df, class_names, model_path, report_name)

  # Makes markdown report using the plots and stuff
  from scripts.model_reporting import make_md_notes
  make_md_notes(model, df, report_name, class_names, model_path)



# progress bar
def progress(count, total, status=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
  sys.stdout.flush()

"""
Itterates through all test images, prints predictions, confidence levels 
and wheather it actually  predicted it correctly it should return a pandas dataframe
 """
def test_all_imgs(model, class_names, test_data_path, size):

  pandas_data = []
  # makes list of test data paths
  data_paths = []
  for root, dirs, files in os.walk(test_data_path):
    for name in files:
      data_paths.append(os.path.join(root, name))

  # Makes preditctions of every image in the data paths list
  print("Running Tests")
  for count, img_path in enumerate(data_paths):

    progress(count,len(data_paths))

    # make_activation_map(model,img_path, class_names) # Makes activation maps
    temp_data = []
    data = np.ndarray(shape=(1, size[0], size[1], 3), dtype=np.float32)
    image = Image.open(img_path)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    normalized_image_array = image_array.astype(np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    label_prediction = np.argmax(prediction[0])

    test_file_name = re.findall("\/.*\/(.*\.jpg)", img_path)
    parent_dir = re.findall("\/.*\/(.*)\/.*\/*.jpg", img_path)

    # find the parent directory applying it to the rubric then get the index of the class name
    prediction_truth_index = class_names.index(parent_dir[0])
    prediction_truth = class_names[prediction_truth_index]

    if class_names[label_prediction] == prediction_truth:

      temp_data.append("True")
    else:
      temp_data.append("False")

    temp_data.append(class_names[label_prediction]) 
    temp_data.append(prediction_truth) 
    # temp_data.append(100 * np.max(prediction[0])) 
    temp_data.append(np.max(prediction[0])) 
    temp_data.append(f"{parent_dir[0]}/{test_file_name[0]}")

    pandas_data.append(temp_data)
  return pandas_data 






if __name__ == '__main__':

  import uuid

  test_data_path = "datasets/testing/"
  class_names = ["AllWires", "BrokenWires", "FooBar", "Glue", "NoWires", "OneThirdsWires", "TwoThirdsWires"]
  defualt_report_name = str(uuid.uuid1())
  default_size=(480,480)

  parser = argparse.ArgumentParser(description='Program to test model against testing dataset')
  parser.add_argument("-c", "--classnames", action="store", type=list, default=class_names)
  parser.add_argument("-t", "--testdatapath", action="store", type=str, default=test_data_path)
  parser.add_argument("-r", "--reportname", action="store", type=str, default=defualt_report_name)
  parser.add_argument("-s", "--inputsize", action="store",type=tuple, default=default_size)
  parser.add_argument("-p", "--modelpath", action="store",type=str, required=True)

  args = parser.parse_args()

  subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)

  main(args)

