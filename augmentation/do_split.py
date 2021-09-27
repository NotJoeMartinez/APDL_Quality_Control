import os, sys, shutil, subprocess, random, math, glob 
from skimage.transform import rotate 
from skimage import img_as_ubyte
import cv2
import numpy as np
from skimage import io

def do_split(directory):
    tree_dict = aug.get_tree_dict(directory) 

    for sub_dir in tree_dict:
        
        # finds 30% of the length of images in directory
        thirty_percent = math.floor((len(tree_dict[sub_dir]) / 100) * 30) 

        move_dict = aug.parse_tree_dict(tree_dict,sub_dir,thirty_percent)
        return move_dict
