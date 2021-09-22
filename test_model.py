import os, sys, random, re, subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 
from os import path
from PIL import Image, ImageOps
from tensorflow import keras

subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)
test_data_path = "datasets/testing/"
class_names = ["AllWires", "BrokenWires", "FooBar", "Glue", "NoWires", "OneThirdsWires", "TwoThirdsWires"]

def main(class_names=class_names, test_data_path=test_data_path, report_name=sys.argv[2], size=(480,480)):
    # loads model if user supplied path
    try: 
      model_path = sys.argv[1]
      model = keras.models.load_model(f"{model_path}") 
    except IndexError:
      print("You need to specifiy a model")
    
    model.summary()

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
    plot_confusion_matrix(df,fig_name=f"notes/imgs/{report_name}.png", show=False)  

    # for random sampleing 
    random_test_plot(model, class_names, test_data_path, report_name, size, show=False)

    # for calulating results
    from scripts.model_reporting import calculate_results
    calculate_results(df, class_names, model_path, report_name)

    # Makes markdown report using the plots and stuff
    from scripts.model_reporting import make_md_notes
    make_md_notes(model, df, report_name, class_names, model_path)


""" Plots a confusion matrix and saves it to notes/imgs/{model_name.png}"""
def plot_confusion_matrix(df, fig_name="", show=False):
    import seaborn as sn
    fig, ax = plt.subplots(figsize=(20, 10))
    confusion_matrix = pd.crosstab(df['predicted'], df['actual'], rownames=['Predicted'], colnames=['Actual'])

    sn.heatmap(confusion_matrix, annot=True, ax=ax)
    plt.savefig(fig_name)

    if show == True:
      plt.show()


""" Plots prediction in a bar chart format with the red bar being the best guess"""
def plot_value_array(predictions_array, class_names):
    plt.grid(False)
    plt.xticks(range(7))
    plt.yticks([])
    thisplot = plt.bar(range(7), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')


  
""" Plots the image in which the model is making predictions on and the percentage certanty """
def plot_image(predictions_array, class_names, img_array):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_array, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                  100*np.max(predictions_array)))


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


""" Builds the plot with images of random images and one image of a broken wire """
def random_test_plot(model, class_names, test_data_path, report_name, size, show=False):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    data_paths = []
    for root, dirs, files in os.walk(test_data_path):
        for name in files:
            data_paths.append(os.path.join(root, name))

    random_test_images = random.choices(data_paths, k=6)
    num_rows = 3
    num_cols = 3
    num_images = num_rows*num_cols

    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i, img_path in enumerate(random_test_images):
        data = np.ndarray(shape=(1, size[0], size[1], 3), dtype=np.float32)
        image = Image.open(img_path)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        normalized_image_array = image_array.astype(np.float32)
        data[0] = normalized_image_array
        prediction = probability_model.predict(data)

        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(prediction[0], class_names, image_array)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(prediction[0], class_names)
        _ = plt.xticks(range(7), class_names, rotation=90)

    plt.tight_layout()

    plt.savefig(f"notes/imgs/rand_samples_{report_name}.png")
    if show == True:
      plt.show()



if __name__ == '__main__':
    main()
    sys.exit(0)
