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


import logging
time_stamp = dt.datetime.now().strftime("%H_%M_%S%p")
logging_file = f"logs/{time_stamp}.log"
print(f"logging file: {logging_file}")
logging.basicConfig(filename=f'{logging_file}', level=logging.DEBUG)


def main(args):
    original_dir = args.original_dir
    max_training = args.max_training_number
    max_testing = args.max_testing_number

    dirpaths = get_dir_paths(original_dir)

    # you dont need to call this twice because copytree makes 
    os.makedirs(dirpaths["testing"], exist_ok=False)
    shutil.copytree(original_dir,dirpaths["training"])

    aug.do_split(original_dir, dirpaths)

    aug.augment_data(f"{dirpaths['training']}", max_training, 'edge')
    aug.augment_data(f"{dirpaths['testing']}", max_testing, 'edge' )



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