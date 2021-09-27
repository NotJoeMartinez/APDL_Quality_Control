import os, math, sys
import argparse
import inspect
import random
import shutil
import datetime as dt
import re
# take existing dataset and randomly pick two images from each sub directory
# move these images to a new traing directory preserving their sub directories
# Augment the remaining images and make sure you're not messing that up 

def do_split(directory):
    """[summary]
    Trains calculates 30% of the total images

    Args:
        directory (string): the training directory is supplied to this as an arg

    """
    tree_dict = get_tree_dict(directory) 

    for sub_dir in tree_dict:
        
        # finds 30% of the length of images in directory
        thirty_percent = math.floor((len(tree_dict[sub_dir]) / 100) * 30) 

        move_dict = parse_tree_dict(tree_dict,sub_dir,thirty_percent)
        mv_train_dirs(directory, move_dict)


''' Gets dictionary of filetree ''' 
def get_tree_dict(root_path):
    tree_dict = {}
    sub_dirs = [dirname.name for dirname in os.scandir(root_path)] 

    # Add sub dirs to tree dictionary
    for sub_dir in sub_dirs:
        files = os.listdir(root_path + "/" + sub_dir) 

        tree_dict[sub_dir] = files
    
    return tree_dict


def parse_tree_dict(tree_dict, key, number_to_grab):
    move_dict = {}
    # r_list = random.choices(tree_dict[key], k=number_to_grab)
    foo = tree_dict[key]
    random.shuffle(foo)
    r_list = foo[:number_to_grab]

    move_dict[key] = r_list
    return move_dict


"""
Moves stuff to testing directory
"""
def mv_train_dirs(root_path, move_dict):

    move_list = []
    for key in move_dict:
        for index in move_dict[key]:
            move_list.append(f"{key}/{index}")

    for path in move_list:
        parent_dir = re.findall("^(\w*)\/", path)
        new_dir = f"datasets/testing/{parent_dir[0]}"
        os.makedirs(new_dir, exist_ok=True)
        src = f"{root_path}/{path}"
        shutil.move(src, new_dir)


