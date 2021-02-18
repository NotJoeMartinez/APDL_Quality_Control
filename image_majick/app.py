import os, random
import numpy as np


# randomly get grab a cat and use it as the target cat for the rest of the tests

def get_perfect_cat(cat_len=20):
    files = os.listdir('data/cats')
    cats = [i for i in files]
    # suffle cats
    random.shuffle(cats)
    # cut cats to specific range 
    cut_cats = np.arange(0,cat_len) 

    random_index = random.randint(0,len(cats)-1)
    target_cat = cats[random_index]
    return 'data/cats/{}'.format(target_cat)


# os.sytem runs shell command 
# macos "open" is similar to debian xdg-open 
perfect_cat = get_perfect_cat()
os.system("open {}".format(perfect_cat))

os.system("compare -density 300 {} {} -compose src {}")



