import cv2
import numpy as np
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise

#Lets define functions for each operation
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

#I would not recommend warp_shifting, because it distorts image, but can be used in many use case like 
#classifying blur and non-blur images
def warp_shift(image): 
    transform = AffineTransform(translation=(0,40))  #chose x,y values according to your convinience
    warp_image = warp(image, transform, mode="wrap")
    return warp_image


#use dictionary to store names of functions 
transformations = {
                      'horizontal flip': h_flip, 
                      'vertical flip': v_flip,
                   'adding noise': add_noise,
                   'blurring image': blur_image
                 }                

images_path=str(input("normal images path: "))
augmented_path=str(input("augmented images path: "))

images=[]  
# read image name from folder and append its path into "images" array     
for im in os.listdir(images_path):  
    images.append(os.path.join(images_path,im))

# you can change this value according to your requirement
images_to_generate=input("number of images to generate (Press enter to use default 200 ): ")

if images_to_generate == "":
    # find the amount of images to generate by subtracting the current amount of image in the directory by 200 
    images_to_generate = 200 - len(images)
    print("augmenting {} images".format(images_to_generate))


i = 1                        
while i <= images_to_generate:    
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

    #randomly choosing method to call
    while n <= transformation_count:
        key = random.choice(list(transformations)) 
        transformed_image = transformations[key](original_image)
        n = n + 1
        
    new_image_path = "{}augmented_image_{}.png".format(augmented_path, i)
    # Convert an image to unsigned byte format, with values in [0, 255].
    transformed_image = img_as_ubyte(transformed_image)  
    # convert image to RGB before saving it
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) 
    # save transformed image to path
    cv2.imwrite(new_image_path, transformed_image) 
    i = i+1
