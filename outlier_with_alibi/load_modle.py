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
from alibi_detect.utils.saving import save_detector, load_detector

import sys




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

# test images path
# path_test = "imgs/capsule/test//**/*.*"  

# calls img_to_np
test = img_to_np(path_test)
test = test.astype('float32') / 255.

# loads pretrained model
od = load_detector("models/2346/")
od.infer_threshold(test, threshold_perc=95)

preds = od.predict(test, outlier_type='instance',
            return_instance_score=True,
            return_feature_score=True)
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

plot_feature_outlier_image(preds, test, 
                           X_recon=recon,  
                           max_instances=5,
                           outliers_only=True,
                           figsize=(15,15))


