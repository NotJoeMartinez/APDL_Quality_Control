import cv2
import numpy as np
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise
import argparse
import sys
import tempfile
import glob
import shutil
from pathlib import Path
import subprocess
from scripts import separate_datasets

def main():

    # Remove all dotfiles from currend dir 
    subprocess.run("find datasets -type f -name '\.*' -delete", shell=True)
    subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)
    

    # copies original dataset to training dataset
    subprocess.run("cp -r datasets/original/ datasets/training/", shell=True)

    root_dir = "datasets/training/" 

    separate_datasets.main()


    if not os.path.isdir(root_dir):
        print('The path specified does not exist')
        sys.exit()
    else:
        target_dirs = os.listdir(root_dir) 
    
    for sub_dir in target_dirs: 
        print(sub_dir)
        images_path = f"{root_dir}/{sub_dir}"
        augment(images_path)




def augment(images_path):
    # use dictionary to store names of functions 
    transformations = {
                        'horizontal flip': h_flip, 
                        'vertical flip': v_flip,
                    'adding noise': add_noise,
                    'blurring image': blur_image,
                    'anticlockwise rotation':anticlockwise_rotation, 
                    'clockwise rotation': clockwise_rotation,
                    
                    }                

    augmented_path = f"{images_path}/temp/"

    try: 
        os.mkdir(augmented_path)
    except FileExistsError:
        pass

    images=[]  
    # read image name from folder and append its path into "images" array     
    for im in os.listdir(images_path):  
        images.append(os.path.join(images_path,im))

        images_to_generate = 200 - len(images)


    i = 1                        
    while i <= int(images_to_generate):    
        image = random.choice(images)
        try: 
            original_image = io.imread(image)
        except ValueError:
            pass

        transformed_image = None

        # variable to iterate till number of transformation to apply
        n = 0       
        # choose random number of transformation to apply on the image
        transformation_count = random.randint(1, len(transformations)) 

        # randomly choosing method to call
        while n <= transformation_count:
            key = random.choice(list(transformations)) 
            try: 
                transformed_image = transformations[key](original_image)
            except UnboundLocalError as ue:
                print(ue)
                pass

            n = n + 1
            
        new_image_path = "{}augmented_image_{}.jpg".format(augmented_path, i)
        # Convert an image to unsigned byte format, with values in [0, 255].
        transformed_image = img_as_ubyte(transformed_image)  
        # convert image to RGB before saving it
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) 
        # save transformed image to path
        cv2.imwrite(new_image_path, transformed_image) 
        i = i+1

    for aug_file in glob.glob(f"{augmented_path}*.jpg"):
        shutil.move(aug_file,images_path)

    try:
        os.rmdir(augmented_path)
    except OSError as e:
        print(f'Error: {augmented_path} : {e.strerror}')



def anticlockwise_rotation(image):
    angle = random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle = random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)

# classifying blur and non-blur images
def warp_shift(image): 
    transform = AffineTransform(translation=(0,40))  #chose x,y values according to your convinience
    warp_image = warp(image, transform, mode="wrap")
    return warp_image





if __name__ == '__main__':
    main()
    sys.exit(0)
