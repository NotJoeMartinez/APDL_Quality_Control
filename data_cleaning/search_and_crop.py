import subprocess 
from pathlib import Path

def crop_imgs():    
    root_path = "datasets/uuid_allimgs"
    for path in Path(root_path).rglob('*.jpg'):
        full_path = str(path) 
        # subprocess.run(f"convert -crop 480x480+80+0 {full_path} {full_path}", shell=True)
        subprocess.run(f"file {full_path}", shell=True)

crop_imgs()