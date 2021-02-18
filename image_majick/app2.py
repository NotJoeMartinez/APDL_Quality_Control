# Importing Required Library
from PIL import Image
# Opening Image as an object
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def show_histogram():
    img = Image.open("data/pil/cat1.jpg")
    r, g, b = img.split() 
    len(r.histogram()) 
    ### 256 ### 
    print(r.histogram())
    print(g.histogram())
    print(b.histogram())

show_histogram()

