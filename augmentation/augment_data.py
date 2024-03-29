import os, shutil, random 
from skimage.transform import rotate 
from skimage import img_as_ubyte
import cv2
from skimage import io
from glob import glob
import model_testing
import database as db

def augment_data(original_dir, imgs_per_dir, fill_mode, now):
    transformations = {
                    'anticlockwise_rotation': anticlockwise_rotation, 
                    'clockwise_rotation': clockwise_rotation
                    }                

    for sub_dir in os.listdir(original_dir): 
        print(f"preforming transformations on {sub_dir}\n")
        images_path = f"{original_dir}/{sub_dir}"

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
            model_testing.progress(i, images_to_generate)
            image = random.choice(images)
            original_image = io.imread(image)

            # choose random number of transformation to apply on the image
            transformed_image = None
            transformation_count = random.randint(1, len(transformations)) 

            # variable to iterate till number of transformation to apply
            n=0       
            # randomly choosing method to call
            while n <= transformation_count:
                key = random.choice(list(transformations)) 

                try: 
                    if key == "anticlockwise_rotation" or key == "clockwise_rotation":
                        transformed_image = transformations[key](original_image, fill_mode)[0]
                        angle = transformations[key](original_image, fill_mode)[1]
                        new_image_path = "{}augmented_{}_{}_{}.jpg".format(augmented_path,angle,key,i)
                        db.log_augmentation(image,new_image_path,key,now,angle)

                    else:
                        transformed_image = transformations[key](original_image)
                        new_image_path = "{}augmented_{}_{}.jpg".format(augmented_path,key,i)
                        db.log_augmentation(image,new_image_path,key,now, angle)

                        
                except UnboundLocalError as ue:
                    print(ue)
                    pass
                n+=1
                
            # Convert an image to unsigned byte format, with values in [0, 255].
            transformed_image = img_as_ubyte(transformed_image)  
            # convert image to RGB before saving it
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) 
            # save transformed image to path
            cv2.imwrite(new_image_path, transformed_image) 
            i += 1

        for aug_file in glob(f"{augmented_path}*.jpg"):
            shutil.move(aug_file,images_path)

        try:
            os.rmdir(augmented_path)
        except OSError as e:
            print(f'Error: {augmented_path} : {e.strerror}')


def anticlockwise_rotation(image, fill_mode):
    angle = random.randint(0,180)
    return [rotate(image, angle, resize=False,  cval=0, mode=fill_mode), angle]

def clockwise_rotation(image, fill_mode):
    angle = random.randint(0,180)
    return [rotate(image, -angle, resize=False, cval=0,  mode=fill_mode), angle]

