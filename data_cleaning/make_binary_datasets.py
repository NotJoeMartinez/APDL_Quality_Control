import os, json, subprocess, re
from os import listdir
from os.path import isfile, join
import pathlib, shutil



LABEL_DIR_DICT = {
    "FooBar": "FooBar",
    "Bonded": "Bonded",
    "Unbonded": "UnBonded"
    }

IMAGES_PATH = 'data/original_imgs'
JSON_PATH = 'final_bonded_unbonded.json'

def main(label_dir_dict=LABEL_DIR_DICT, images_path=IMAGES_PATH, json_path=JSON_PATH):

    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    label_dict = make_label_dict(only_files, json_path)
    make_sorted_directories(label_dir_dict)
    copy_dataset(label_dict, only_files, label_dir_dict)
   
def make_sorted_directories(label_dir_dict):
    class_names = [key for key in label_dir_dict]

    pathlib.Path("data").mkdir(parents=True, exist_ok=True) 
    for class_name in class_names:
        pathlib.Path(f"data/sorted_data/{class_name}").mkdir(parents=True, exist_ok=True) 


def copy_dataset(label_dict, only_files, label_dir_dict):

    for target_dir in label_dir_dict:

        for image_name in only_files:
            print(only_files, label_dict)
            image_path = f"data/original_imgs/{image_name}"
            print(target_dir, label_dict[image_name])
            try: 

                if target_dir == label_dict[image_name]:
                    shutil.copy(image_path, f'data/sorted_data/{target_dir}')
                else:
                    shutil.copy(image_path, f'data/sorted_data/{target_dir}')
            except KeyError:
                    shutil.copy(image_path,'data/sorted_data/')
                    pass


def make_label_dict(only_files, json_path):
    label_dict = {}
    with open(json_path, "r") as read_json:
        data = json.load(read_json)
        for i in data:
            try:
                labels = i['annotations'][0]['result'][0]['value']['choices']
                image_name = i['file_upload']
                
                try:
                    image_name = re.search('^[^_]+(?=_)', image_name).group()
                except AttributeError:
                    pass

                if image_name.endswith(".jpg") == False:
                    image_name = image_name + ".jpg"

                label_dict.update({image_name:labels})
            except IndexError:
                pass
    return label_dict

if __name__ == '__main__':
    main()

