# image_diff
Given hundreds of similar images image_diff finds the images that match the "perfect" image. This project is intended to help with the quality control process of manufacturing 100s chipsets.


## Data Set 
In order to do this I'm first going to use a data set of 12,500 cat images from [this](https://github.com/ADlead/Dogs-Cats) repo that was intended to train 
a machine learning model to identify cats & dogs. Eventually I need to find a way to make this more realistic by using an existing dataset of chipsets that intentionally contains flawed images. 

## ImageMajick

I shopped a minor difference in the same rasberi pi 

![README.assets/image-20210125223744102](/Users/supernova/labwork/image_diff/README.assets/image-20210125223744102.png)

ImageMajick diff code 
```bash
compare -density 300  pi1.jpg pi2.jpg -compose src pi_diff.jpeg
```
Result 


![image-20210125222803849](/Users/supernova/labwork/image_diff/README.assets/image-20210125222803849.png)

## Tool Set 

- Python
- Open CV
- Tinker
- Python Imaging Library (PIL or Pillow)
