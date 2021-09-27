import os, sys, shutil, subprocess, random, math, glob 
from skimage.transform import rotate 
from skimage import img_as_ubyte
import cv2
import numpy as np
from skimage import io
import augmentation as aug


def main(args):
    original_dir = args.original_dir
    original_dir = args.training_dir
    testing_dir = args.testing_dir
    max_testing = args.max_testing_number
    max_training = args.max_training_number

    os.makedirs(testing_dir, exist_ok=False)
    shutil.copytree(original_dir,training_dir)

    aug.do_split(original_dir)
    augment_data(training_dir, max_training, 'edge')
    augment_data(testing_dir, max_testing, 'edge' )


def augment_data(root_dir, imgs_per_dir, fill_mode):
    transformations = {
                    'horizontal flip': h_flip, 
                    'vertical flip': v_flip,
                    'anticlockwise rotation':anticlockwise_rotation, 
                    'clockwise rotation': clockwise_rotation,
                    
                    }                
    for sub_dir in os.listdir(root_dir): 
        images_path = f"{root_dir}/{sub_dir}"

        # make a temp directory for the augmented images so you're not augmenting 
        augmented_path = f"{images_path}/temp/"
        try: 
            os.mkdir(augmented_path)
        except FileExistsError:
            pass

        # read image name from folder and append its path into "images" array     
        images=[]  
        for im in os.listdir(images_path):  
            images.append(os.path.join(images_path,im))
        
        images_to_generate = imgs_per_dir - len(images)

        # remove this from imgs array because it's a directory not an image
        images.remove(augmented_path[:-1])

        i = 1 
        while i <= int(images_to_generate):    
            image = random.choice(images)
            original_image = io.imread(image)
            transformed_image = None
            # choose random number of transformation to apply on the image
            transformation_count = random.randint(1, len(transformations)) 

            # variable to iterate till number of transformation to apply
            n = 0       
            # randomly choosing method to call
            while n <= transformation_count:
                key = random.choice(list(transformations)) 
                try: 
                    transformed_image = transformations[key](original_image, fill_mode)

                except UnboundLocalError as ue:
                    print(ue)
                    pass
                n = n + 1
                
            new_image_path = "{}augmented_image_{}.jpg".format(augmented_path, i)
            print(new_image_path)
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


def anticlockwise_rotation(image, fill_mode):
    angle = random.randint(0,180)
    return rotate(image, angle, resize=False,  cval=0, mode=fill_mode)

def clockwise_rotation(image, fill_mode):
    angle = random.randint(0,180)
    return rotate(image, -angle, resize=False, cval=0,  mode=fill_mode)

def h_flip(image, fill_mode):
    return  np.fliplr(image)

def v_flip(image, fill_mode):
    return np.flipud(image)


if __name__ == '__main__':

    subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)

    original_dir = "datasets/original"
    testing_dir =  "datasets/testing"
    training_dir =  "datasets/training"
 
    import argparse

    parser = argparse.ArgumentParser(description='Program to augment original dataset')

    parser.add_argument("-od","--original-dir", action="store", type=str, required=True, 
                        help="Original directory of the data, make sure you want to do this" )
    parser.add_argument("-trdir", "--training-dir", action="store",type=str, default="datasets/training",
                        help="Directory of training data, defaults to datasets/training" )
    parser.add_argument("-tsdir", "--testing-dir", action="store",type=str, default="datasets/testing",
                        help="Directory of testing data, defaults to datasets/testing" )
    parser.add_argument("-mtrain", "--max-training-number", action="store", type=int, default=566,
                        help="Augment training images up to this number")
    parser.add_argument("-mtest", "--max-testing-number", action="store", type=int, default=176,
                        help="Augment testing images up to this number")

    args=parser.parse_args()

    main(args)