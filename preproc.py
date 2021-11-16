import pandas as pd
import numpy as np
from PIL import Image
import pygit2
import os

path = os.getcwd()+'/Fruit'
if os.path.exists(path):
    # do nothing
    print("Already exists")
else:
    cloned = pygit2.clone_repository("https://github.com/ReeseReynolds/ML-Testing", path, bare=False, repository=None, remote=None, checkout_branch=None, callbacks=None)

# gather all class labels and file names
img_set = []
for root, dirs, files in os.walk(path):
    if 'Train' in root:
        for file in files:
            p = root.replace(path,'')
            p = p.replace('Train', '')
            p = p.replace(str(file), '')
            label = p.replace('\\', '')
            img_set.append([root+'/'+file, label, file])

# output dir for testing
if not os.path.exists(os.getcwd()+'/out'):
    os.mkdir(os.getcwd()+'/out')

# calc file size of train folder
file_count = sum(len(files) for _, _, files in os.walk(path+'/Train'))
print(file_count)

# loop through training set and create np array of pixel values
for i in range(file_count):
    fname = img_set[i][0]
    im = Image.open(fname, 'r').convert('RGB')
    pix = list(im.getdata())
    im.close()

    dt = np.dtype('uint8','uint8','uint8')
    pix_arr = np.array(pix, dtype=dt)
    pix_fin = np.reshape(pix_arr, (100, 100, 3))

    # print(len(pix_arr))
    # print(pix_arr)

    # o_name = "out/out" + str(i) + ".jpg"
    # imageio.imwrite(o_name, pix_fin)