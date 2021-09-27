import os, random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import tensorflow as tf

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