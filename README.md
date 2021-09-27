# APDL_Quality_Control
This project is intended to help with the quality control process of manufacturing calorimeters. 

### Installation 
It's important to specify this version of python as Tenserflow has not been working on the latest release 
```bash
git clone https://github.com/NotJoeMartinez/APDL_Quality_Control
cd APDL_Quality_Control 
virtualenv env --python=python3.9.2
source env/bin/activate 
pip install -r requirements.txt
```

If you still get an error installing tenserflow deactivate and remove
```bash
pip install --upgrade tensorflow
```

### augment_imgs.py
- Augments the original data
- Handles splitting the dataset up into training and testing. 
- Randomly applies a rotation function to data  

### train_model.py
- Trains model on the augmented dataset 
- Creates a plot of the training history 
- Writes some metadata to a csv

### test_model.py
- Tests trained models against the testing dataset
- Plots data into confusion matrix along with ac couple other visualizations
- Creates a markdown file with impeded plots and other metadata

### transfer_learning.py 
- Trains model using transfer learning model "mobilenet_v3_small_100_224"
- Not fully tested to current workflow (Mon Sep 27 07:37:26 CDT 2021)