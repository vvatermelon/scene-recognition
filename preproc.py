import imageio
import pandas as pd
import numpy as np
import time
from PIL import Image
import pygit2
import copy
import os


# topology:
#   input layer of 1            - one rgb image
#   first conv layer of 4       - four convolutions for features: edges, average color, texture detection, size detection
#   pooling layer of 4          - all four features can be extracted from the same 50x50x3 images but maybe rgb can be converted into black and white instead?
#   second conv layer of 16     - four new filters to apply to all previous four images which should evolve with weights
#   second pooling layer of 16  - pool all sixteen images
#   3 hidden layers of 16       - 3 weighted layers of 16 neurons each with corresponding weights in the weights matrix
#   output layer of 131         - 131 output neurons can be stored as list[] * 131; each output neuron should receive input from each neuron in the final hidden layer

class CNN:
    def __init__(self, layers, epochs):
        self.classes = ["Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1", "Apple Red 2", "Apple Red 3", "Apple Red Delicious", "Apple Red Yellow 1", "Apple Red Yellow 2", "Apricot", "Avocado", "Avocado ripe", "Banana", "Banana Lady Finger", "Banana Red", "Beetroot", "Blueberry", "Cactus fruit", "Cantaloupe 1", "Cantaloupe 2", "Carambula", "Cauliflower", "Cherry 1", "Cherry 2", "Cherry Rainier", "Cherry Wax Black", "Cherry Wax Red", "Cherry Wax Yellow", "Chestnut", "Clementine", "Cocos", "Corn", "Corn Husk", "Cucumber Ripe", "Cucumber Ripe 2", "Dates", "Eggplant", "Fig", "Ginger Root", "Granadilla", "Grape Blue", "Grape Pink", "Grape White", "Grape White 2", "Grape White 3", "Grape White 4", "Grapefruit Pink", "Grapefruit White", "Guava", "Hazelnut", "Huckleberry", "Kaki", "Kiwi", "Kohlrabi", "Kumquats", "Lemon", "Lemon Meyer", "Limes", "Lychee", "Mandarine", "Mango", "Mango Red", "Mangostan", "Maracuja", "Melon Piel de Sapo", "Mulberry", "Nectarine", "Nectarine Flat", "Nut Forest", "Nut Pecan", "Onion Red", "Onion Red Peeled", "Onion White", "Orange", "Papaya", "Passion Fruit", "Peach", "Peach 2", "Peach Flat", "Pear", "Pear 2", "Pear Abate", "Pear Forelle", "Pear Kaiser", "Pear Monster", "Pear Red", "Pear Stone", "Pear Williams", "Pepino", "Pepper Green", "Pepper Orange", "Pepper Red", "Pepper Yellow", "Physalis", "Physalis with Husk", "Pineapple", "Pineapple Mini", "Pitahaya Red", "Plum", "Plum 2", "Plum 3", "Pomegranate", "Pomelo Sweetie", "Potato Red", "Potato Red Washed", "Potato Sweet", "Potato White", "Quince", "Rambutan", "Raspberry", "Redcurrant", "Salak", "Strawberry", "Strawberry Wedge", "Tamarillo", "Tangelo", "Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato Cherry Red", "Tomato Heart", "Tomato Maroon", "Tomato not Ripened", "Tomato Yellow", "Walnut", "Watermelon"]
        self.neurons = 48
        layers = layers
        train_count, test_count, path = self.getRepo()
        train_data, test_data = self.getData(path)
        self.fit(train_count, train_data, epochs)


    def getRepo(self):
        path = os.getcwd() + '/Fruit'
        if os.path.exists(path):
            # do nothing
            print("Already exists")
        else:
            cloned = pygit2.clone_repository("https://github.com/ReeseReynolds/ML-Testing", path, bare=False,
                                             repository=None, remote=None, checkout_branch=None, callbacks=None)

        # temp dir for testing
        if not os.path.exists(os.getcwd() + '/out'):
            os.mkdir(os.getcwd() + '/out')

        # calc file size of train folder
        train_count = sum(len(files) for _, _, files in os.walk(path + '/Train'))
        print(train_count)

        # calc file size of test folder
        test_count = sum(len(files) for _, _, files in os.walk(path + '/Test'))
        print(test_count)
        return train_count, test_count, path

    def getData(self, path):
        # gather all class labels and file names
        train_set = []
        test_set = []
        lizt = []
        for root, dirs, files in os.walk(path):
            if 'Train' in root:
                for file in files:
                    p = root.replace(path, '')
                    p = p.replace('Train', '')
                    p = p.replace(str(file), '')
                    label = p.replace('\\', '')
                    train_set.append([root + '/' + file, self.classes.index(label), file])
            if 'Test' in root:
                for file in files:
                    p = root.replace(path, '')
                    p = p.replace('Test', '')
                    p = p.replace(str(file), '')
                    label = p.replace('\\', '')
                    test_set.append([root + '/' + file, self.classes.index(label), file])
        f = open("C:/Users/Owner/Desktop/list.txt", "a")
        for x in lizt:
            f.write("\"" + x + "\", ")
        f.close()
        print(len(lizt))
        return train_set, test_set

    def genSlices(self, raw_im):
        slices = []
        # slices[0-3] generate and store 4 50x50x3 images from all quadrants of raw_im (RGB so there are 3 color channels)
        # TODO implement slice function - should be similar to code in fit function
        return slices

    def relu(self, x):
        # TODO simple max ReLU function to be called after each layer
        return np.maximum(0, x)

    def train(self, slices, weights):
        # create all layers, some should require extra parameters to be implemented
        # TODO implement backpropagation

            # input into convolution layer 1
            # ReLu output of conv layer 1
            # pool outputs

            # input pool into convolution layer 2
            # ReLu output of conv layer 2
            # pool outputs

            # nn strongly connected layers

        return weights

    def fit(self, file_count, img_set, epochs):
        # loads and trains with 10k images at a time until all images have been iterated through
        # generate weights randomly, 16 weights for each layer
        weights = [[np.random() for i in 16] for j in range(self.layers)]
        for e in range(epochs):
            x_data = []
            start = time.time()
            for i in range(file_count):
                # print(img_set[i])
                # get image
                fname = img_set[i][0]
                im = Image.open(fname, 'r')
                pix = list(im.getdata())
                im.close()
                # x_data.append(pix) # is this necessary?

                # generates textured and outlined image by coloring adverse changes with magenta (255,0,255)
                # good reference for convolutions and slicing images
                edge = copy.deepcopy(pix)
                for p in range(len(pix) - 3):
                    scene = pix[p:p + 3]
                    subd = tuple(map(lambda x, y: x - y, scene[2], scene[0]))
                    for x in subd:
                        if x > 40 or x < -40:
                            edge[p * 1] = (255, 0, 255)

                dt = np.dtype('uint8', 'uint8', 'uint8')
                pix_arr = np.array(edge, dtype=dt)
                pix_fin = np.reshape(pix_arr, (100, 100, 3))
                # o_name = "out/out" + str(i) + ".jpg"
                # imageio.imwrite(o_name, pix_fin)
                im_slices = self.genSlices(pix_fin)
                weights = self.train(im_slices, weights)


                if (i+1) % 11282 == 0 and (i+1) != 0:
                    print("1/6 total images processed")
                    print(time.time() - start, " seconds for i = ", i)
                    start = time.time()
                    # x_data.clear()

p = CNN(2,5)