import os, random, uuid, sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import tensorflow as tf
import cv2
import seaborn as sn

""" Plots a confusion matrix and saves it to notes/imgs/{model_name.png}"""
def plot_confusion_matrix(df, fig_name="", show=False):
    print(df)
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

  

  """[summary]
    Makes activation maps 
  """
def make_activation_map(model, img_path, class_names):

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    DIM = 480

    data = np.ndarray(shape=(1, DIM, DIM, 3), dtype=np.float32)
    img = Image.open(img_path)
    img = ImageOps.fit(img, (DIM, DIM), Image.ANTIALIAS)
    image_array= np.asarray(img)
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    normalized_image_array = image_array.astype(np.float32)
    data[0] = normalized_image_array
    prediction = probability_model.predict(data)
    label_prediction = class_names[np.argmax(prediction[0])]
    plt.xlabel("Predicted Label: {} \n Confidence: {:2.0f}%".format(label_prediction, 100*np.max(prediction[0])))

    
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_13')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(data)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)    

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((7, 7))

    img = cv2.imread(img_path )

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img_name = f"notes/activation_maps/superimposed/{str(uuid.uuid1())}.jpg"

    cv2.imwrite(f'{superimposed_img_name}', superimposed_img)
    plt_img = cv2.mpimg.imread(superimposed_img_name)

    plt.imshow(plt_img)
    plt.tight_layout()
    plt.savefig(f"notes/activation_maps/plots/{str(uuid.uuid1())}.png")
    plt.show()



"""[summary]
  shows training details like validation loss based on history
"""
def show_model_details(save_dir, history, epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')

  plt.savefig(f'{save_dir}/performance.png')  


# progress bar
def progress(count, total, status=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
  sys.stdout.flush()