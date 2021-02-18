# https://www.mvtec.com/company/research/datasets/mvtec-ad/
# https://pypi.org/project/alibi-detect/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
# Logging
import logging 
logging.basicConfig(filename='foo.log' , level=logging.DEBUG)
import datetime as dt
# saving model
from alibi_detect.utils.saving import save_detector

def img_to_np(path, resize = True):  
    img_array = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        # convert image to rgb format from png? what is the diff?
        img = Image.open(fname).convert("RGB")
        # image resizing is set to True by default
        if(resize): img = img.resize((64,64)) 
        # How does one append an image to an array? I assume this has something to do with asarray
        img_array.append(np.asarray(img))

    images = np.array(img_array) 
    return images
  
path_train = "imgs/capsule/train//**/*.*" #  old file pattern \\**\*.*
path_test = "imgs/capsule/test//**/*.*"  

train = img_to_np(path_train)
test = img_to_np(path_test)

train = train.astype('float32') / 255.
test = test.astype('float32') / 255.

logging.info('Train Varable:  {}'.format(train, type(train)))
# logging.info('Train[0] Varable: {} '.format(train[0]))
# logging.info('Train[0].shape Varable: {}'.format(train[0].shape))


# Training of model

encoding_dim = 1024
dense_dim = [8, 8, 128]

encoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=train[0].shape),
      Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(encoding_dim,)
  ])

decoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=(encoding_dim,)),
      Dense(np.prod(dense_dim)),
      Reshape(target_shape=dense_dim),
      Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
  ])

od = OutlierAE( threshold = 0.001,
                encoder_net=encoder_net,
                decoder_net=decoder_net)

adam = tf.keras.optimizers.Adam(lr=1e-4)

# save to filepath
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=models_filepath, verbose=1, save_best_only=True)

# Training
print("training")
od.fit(train, epochs=100, verbose=True,
       optimizer = adam)
# saving? 
now = dt.datetime.now()
save_detector(od,"models/{}/".format(now.strftime("%-M%S")))
od.infer_threshold(test, threshold_perc=95)

# outlier score
print("predicting outlier")
preds = od.predict(test, outlier_type='instance',
            return_instance_score=True,
            return_feature_score=True)


print("coppying")
for i, fpath in enumerate(glob.glob(path_test)):
    if(preds['data']['is_outlier'][i] == 1):
        source = fpath
        shutil.copy(source, 'imgs/outliers/')
        
filenames = [os.path.basename(x) for x in glob.glob(path_test, recursive=True)]

dict1 = {'Filename': filenames,
     'instance_score': preds['data']['instance_score'],
     'is_outlier': preds['data']['is_outlier']}
     
df = pd.DataFrame(dict1)
df_outliers = df[df['is_outlier'] == 1]

print(df_outliers, "is outlier")


recon = od.ae(test).numpy()

# plot_feature_outlier_image(preds, test, 
#                            X_recon=recon,  
#                            max_instances=5,
#                            outliers_only=True,
#                            figsize=(15,15))
