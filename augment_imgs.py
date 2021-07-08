import os, sys, shutil, subprocess, random, math, glob
from skimage.transform import rotate 
from skimage import img_as_ubyte
import cv2
import numpy as np
from skimage import io
from scripts.separate_datasets import get_tree_dict, parse_tree_dict, mv_train_dirs 

subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)

def main(original_dir=sys.argv[1]):
    testing_dir =  "datasets/testing"
    training_dir =  "datasets/training"
 

    # os.makedirs(testing_dir, exist_ok=True)
    # shutil.copytree(original_dir,training_dir)
    # do_split(training_dir)
    augment_traing_data(testing_dir, 170)

# split 30% of dataset?
def do_split(directory):
    tree_dict = get_tree_dict(directory) 

    for sub_dir in tree_dict:
        
        # finds 30% of the length of images in directory
        thirty_percent = math.floor((len(tree_dict[sub_dir]) / 100) * 30) 

        move_dict = parse_tree_dict(tree_dict,sub_dir,thirty_percent)

        mv_train_dirs(directory, move_dict)



def augment_traing_data(root_dir, imgs_per_dir):
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
                    transformed_image = transformations[key](original_image)

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




def anticlockwise_rotation(image):
    angle = random.randint(0,180)
    return rotate(image, angle, resize=False,  cval=1, mode='edge')

def clockwise_rotation(image):
    angle = random.randint(0,180)
    return rotate(image, -angle, resize=False, cval=1,  mode='edge')

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)


if __name__ == '__main__':
    main()