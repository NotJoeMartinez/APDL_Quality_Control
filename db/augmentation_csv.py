import csv, datetime, uuid, os

def log_augmentation(og_filepath, new_name, aug_function, now, angle=""):

    # The data assigned to the list 
    list_data=[og_filepath,new_name,aug_function,angle, ]
  

    with open(f'db/csvs/augmentations/{now}.csv', 'a', newline='') as f_object:  
        writer_object = csv.writer(f_object)
        writer_object.writerow(list_data)  
        f_object.close()

