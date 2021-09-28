import os, sys, re, subprocess, argparse
import numpy as np
import pandas as pd 
from PIL import Image, ImageOps
from tensorflow import keras
import model_testing 

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
  if os.path.exists(f"notes/{report_name}.md"):
    overwrite = input("MODEL REPORT EXISTS; Do you want to overwrite? (Y/n): ")
    if overwrite == "n":
      print("Exiting")
      sys.exit() 
    else:
      pass

  # where the testing actually happens
  tested_images = model_testing.test_all_imgs(model, class_names, test_data_path, size) 
  df = pd.DataFrame(tested_images, columns = ['score','predicted','actual','confidence','path'])
  df.to_csv(f'notes/csvs/{report_name}.csv', encoding='utf-8')

  # for confusion matrix
  model_testing.plot_confusion_matrix(df,fig_name=f"notes/imgs/{report_name}.png", show=False)  

  # for random sampleing 
  model_testing.random_test_plot(model, class_names, test_data_path, report_name, size, show=False)

  # for calulating results
  from model_testing.model_reporting import calculate_results
  calculate_results(df, class_names, model_path, report_name)

  # Makes markdown report using the plots and stuff
  from model_testing.model_reporting import make_md_notes
  make_md_notes(model, df, report_name, class_names, model_path)



if __name__ == '__main__':

  import uuid

  test_data_path = "datasets/91021_no_splits/testing/"
  class_names = ["AllWires", "BrokenWires", "FooBar", "Glue", "NoWires", "OneThirdsWires", "TwoThirdsWires"]
  defualt_report_name = str(uuid.uuid1())
  default_size=(480,480)

  parser = argparse.ArgumentParser(description='Program to test model against testing dataset')
  parser.add_argument("-c", "--classnames", action="store", type=list, default=class_names, 
                                            help='Python list of class names ')
  parser.add_argument("-t", "--testdatapath", action="store", type=str, default=test_data_path, 
                                            help="Relative path to testing dataset defualts to datasets/testing/")
  parser.add_argument("-r", "--reportname", action="store", type=str, default=defualt_report_name,
                                            help="Name of the generated markdown report defaults to a uuid")
  parser.add_argument("-s", "--inputsize", action="store",type=tuple, default=default_size,
                                            help="Python tuple of first input layer defaults to (480, 480)")
  parser.add_argument("-p", "--modelpath", action="store",type=str, required=True,
                                            help="Relative path to model, required")

  args = parser.parse_args()

  subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)

  main(args)

