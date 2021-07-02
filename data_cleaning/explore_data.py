import pandas, shutil

csv_data = pandas.read_csv("all_labels.csv")


for i in range(len(csv_data)):
	image_name = f"data/original_imgs/{csv_data.loc[i,'image name']}"
	image_label_path = f"data/sorted_all/{csv_data.loc[i, 'label']}"
	shutil.copy(image_name, image_label_path)
	

