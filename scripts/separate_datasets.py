import os
import sys
import argparse
import inspect
import random
import shutil
import datetime as dt
import re
# take existing dataset and randomly pick two images from each sub directory
# move these images to a new traing directory preserving their sub directories
# Augment the remaining images and make sure you're not messing that up 

def main():
    tree_dict = get_tree_dict("datasets/training/")
    move_dict = parse_tree_dict(tree_dict)
    mv_train_dirs("datasets/training/", move_dict)


def get_tree_dict(root_path):
    tree_dict = {}

    sub_dirs = [dirname.name for dirname in os.scandir(root_path) if dirname.name != '.DS_Store']

    # Add sub dirs to tree dictionary
    for sub_dir in sub_dirs:
        files = os.listdir(root_path + "/" + sub_dir) 

        tree_dict[sub_dir] = files
    
    return tree_dict



def parse_tree_dict(tree_dict):
    move_dict = {}
    for key in tree_dict:
        # make sure you don't get duplicates
        while True:
            r_list = random.choices(tree_dict[key], k=2)
            if r_list[0] == r_list[1]:
                r_list = random.choices(tree_dict[key], k=2)
            else:
                break


                
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
    

    now = dt.datetime.now().strftime("%m_%d_%-I:%M:%S%p")

    for path in move_list:
        parent_dir = re.findall("^(\w*)\/", path)
        # new_dir = f"datasets/testing/{now}/{parent_dir[0]}"
        new_dir = f"datasets/testing/{parent_dir[0]}"
        os.makedirs(new_dir, exist_ok=True)
        dst = new_dir 
        src = f"{root_path}/{path}"
        shutil.move(src, dst)



if __name__ == '__main__':
    main()