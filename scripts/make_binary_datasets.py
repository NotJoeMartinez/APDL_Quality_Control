import os, json, subprocess, re
from os import listdir
from os.path import isfile, join
import pathlib, shutil



LABEL_DIR_DICT = {
    "glue": "Glue",
    "good": "Good",
    "two_third_wire": "Two Third Wire",
    "one_third_wire": "One Third Wire", 
    "unknown_debris": "Unknown Debris",
    "no_wires": "No wires",
    "subtance_on_outer_ring": "Subtance On Outer Ring",
    "strings": "strings",
    "void_glue": "void glue"
}
IMAGES_PATH = 'images'
JSON_PATH = 'project-3-at-2021-06-09-18-40-018e02d0.json'

def main(label_dir_dict=LABEL_DIR_DICT, images_path=IMAGES_PATH, json_path=JSON_PATH):


    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    label_dict = make_label_dict(only_files, json_path)
    make_sorted_directories(label_dir_dict)
    copy_dataset(label_dict, only_files, label_dir_dict)
   
def make_sorted_directories(label_dir_dict):
    class_names = [key for key in label_dir_dict]

    pathlib.Path("data").mkdir(parents=True, exist_ok=True) 
    for class_name in class_names:
        print(class_name)
        pathlib.Path(f"data/{class_name}").mkdir(parents=True, exist_ok=True) 
        pathlib.Path(f'data/{class_name}/no_{class_name}').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(f'data/{class_name}/yes_{class_name}').mkdir(parents=True, exist_ok=True) 


def copy_dataset(label_dict, only_files, label_dir_dict):

    for key in label_dir_dict:
        target_label = label_dir_dict[key]
        target_dir = key
        for image_name in only_files:
            image_path = f"images/{image_name}"
            try: 
                if target_label in label_dict[image_name]:
                    shutil.copy(image_path, f'data/{target_dir}/yes_{target_dir}')
                else:
                    shutil.copy(image_path, f'data/{target_dir}/no_{target_dir}')
            except KeyError:
                    print(image_path, image_name)
                    shutil.copy(image_path, f'data/{target_dir}/no_{target_dir}')
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

