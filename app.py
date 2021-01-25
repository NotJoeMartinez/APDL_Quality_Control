import os, random


# randomly get grab a cat and use it as the target cat for the rest of the tests

def get_perfect_cat():
    files = os.listdir('data/cats')
    cats = [i for i in files]
    random_index = random.randint(0,len(cats)-1)
    target_cat = cats[random_index]
    return 'data/cats/{}'.format(target_cat)


# os.sytem runs shell command 
# macos "open" is similar to debian xdg-open 
perfect_cat = get_perfect_cat()
os.system("open {}".format(perfect_cat))



