import os, sys, shutil, subprocess, random, math
from skimage.transform import rotate 
from skimage import img_as_ubyte
import cv2
import numpy as np
from skimage import io
from glob import glob
from pathlib import Path
import augmentation as aug
import datetime as dt
import database as db
import data_cleaning

def main(args):
    original_dir = args.original_dir
    max_training = args.max_training_number
    max_testing = args.max_testing_number
    now = dt.datetime.now().strftime("%m_%d_%H_%M_%s")

    dirpaths = get_dir_paths(original_dir)


    print("Splitting training data")
    os.makedirs(dirpaths["testing"], exist_ok=False)
    shutil.copytree(original_dir,dirpaths["training"])
    aug.do_split(dirpaths["training"], dirpaths)

    print("augmenting training data")
    aug.augment_data(f"{dirpaths['training']}", max_training, 'edge',now)

    print("augmenting testing data")
    aug.augment_data(f"{dirpaths['testing']}", max_testing, 'edge',now)

    print("Adding csv to database")
    
    db.add_csv_todb(csv_path=f"database/csvs/augmentations/{now}.csv")

    print("Cropping Images")
    data_cleaning.crop_imgs(original_dir)




def get_dir_paths(og_data_path):
    path = Path(og_data_path)
    testing = f"{path.parent.parts[0]}/{path.parent.stem}/testing"
    training = f"{path.parent.parts[0]}/{path.parent.stem}/training"

    dirpaths= {
        "testing": testing,
        "training": training 
    }

    return dirpaths 



if __name__ == '__main__':

    subprocess.run("find . -name '.DS_Store' -type f -delete", shell=True)

 
    import argparse

    parser = argparse.ArgumentParser(description='Program to augment original dataset')

    parser.add_argument("-od","--original-dir", action="store", type=str, required=True, 
                        help="Original directory of the data, make sure you want to do this" )
    parser.add_argument("-mtrain", "--max-training-number", action="store", type=int, default=566,
                        help="Augment training images up to this number")
    parser.add_argument("-mtest", "--max-testing-number", action="store", type=int, default=176,
                        help="Augment testing images up to this number")

    args=parser.parse_args()

    main(args)