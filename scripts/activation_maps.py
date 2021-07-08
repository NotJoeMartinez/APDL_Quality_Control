# imports from tutorial
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# normal imports
import subprocess, os, sys, re, argparse
import datetime as dt
from tensorflow import keras
import uuid
import matplotlib.image as mpimg
from PIL import Image, ImageOps


# subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)
# class_names = ["AllWires", "BrokenWires", "FooBar", "Glue", "NoWires", "OneThirdsWires", "TwoThirdsWires"]
# now = dt.datetime.now().strftime("%m_%d_%I:%M:%S%p")

# def main(class_names=class_names, model_path=sys.argv[1]):
#     # loads model if user supplied path
#     try:
#         model_name = re.search('[^\/]*$', model_path).group()
#         model = keras.models.load_model(f"{model_path}")

#         model.summary()
#     except IndexError:
#         pass
    
#     test_abunch(model, "datasets/testing/")
#     # make_activation_map(model, model_name, img_path)

# def test_abunch(model, img_dir):    
#     data_paths = []
#     for root, dirs, files in os.walk(img_dir):
#         for img in files:
#             data_paths.append(f"{root}/{img}")

#     # print(data_paths)
#     for img_path in data_paths:
#         make_activation_map(model, img_path)




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
    plt_img = mpimg.imread(superimposed_img_name)

    plt.imshow(plt_img)
    plt.tight_layout()
    plt.savefig(f"notes/activation_maps/plots/{str(uuid.uuid1())}.png")
    # plt.show()

   




if __name__ == '__main__':
    main()