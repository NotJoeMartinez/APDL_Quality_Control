import os, math, random, shutil
import datetime as dt
from pathlib import Path

# take existing dataset and randomly pick two images from each sub directory
# move these images to a new traing directory preserving their sub directories
# Augment the remaining images and make sure you're not messing that up 

def do_split(og_directory, dirpaths):
    """[summary]
    Trains calculates 30% of the total images

    Args:
        directory (string): the training directory is supplied to this as an arg

    """

    tree_dict = get_tree_dict(og_directory) 

    for sub_dir in tree_dict:
        
        # finds 30% of the length of images of sub_dir images
        num_imgs_in_sub_dir =len(tree_dict[sub_dir]) 
        thirty_percent = math.floor( (num_imgs_in_sub_dir / 100) * 30)

        # dictonary: key:"parent dir" val:"30% of original images"
        move_dict = parse_tree_dict(tree_dict,sub_dir,thirty_percent)

        mv_train_dirs(og_directory, move_dict, dirpaths)


def get_tree_dict(og_directory):
    """[summary]
    A dictionary with the keys as the classes and the values as the files assoscated with those classes
    Args:
        og_directory (string): original directory passed augment_imgs.py 

    Returns:
        dict: keys->parent dir vales-> files in parent dir  
    """
    tree_dict = {}
    sub_dirs = [dirname.name for dirname in os.scandir(og_directory)] 

    # Add sub dirs to tree dictionary
    for sub_dir in sub_dirs:
        files = os.listdir(og_directory+ "/" + sub_dir) 

        tree_dict[sub_dir] = files
    
    return tree_dict

def parse_tree_dict(tree_dict, sub_dir, number_to_grab):
    """sumary_line
    shuffles the files in the given subdirectory and builds a new dictionary with 30% of the images
    Keyword arguments:
    tree_dict -- description
    key -- 
    Return: dictonary of of file names 
    """

    # Grabs list of files and shuffles list
    shuff_list = tree_dict[sub_dir]
    random.shuffle(shuff_list)

    # grabs only how many files it needs for 30%
    shuff_list = shuff_list[:number_to_grab]

    # appends this list of files to a new dictionary with the parent directory as key
    move_dict = {}
    move_dict[sub_dir] = shuff_list 
    
    return move_dict



def mv_train_dirs(og_directory, move_dict, dirpaths):
    """[summary]
    moves images from move original to testing directory
    """

    for parent_dir in move_dict:
        move_list = move_dict[parent_dir]

        for image in move_list:

            src = f"{og_directory}/{parent_dir}/{image}" 

            dest = str(Path(og_directory).parents[0])
            dest = f"{dest}/testing/{parent_dir}"

            os.makedirs(dest, exist_ok=True)
            shutil.move(src,dest)


