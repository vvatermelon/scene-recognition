import imageio
# import pandas as pd
import numpy as np
import time
from PIL import Image
import pygit2
import copy
import os


# topology:
#   input layer of 1            - one rgb image
#   first conv layer of 4       - four convolutions for features
#   pooling layer of 16          - all four features can be extracted from the same 50x50x3 images but maybe rgb can be converted into black and white instead?
#   second conv layer of 16     - four new filters to apply to all previous 16 images which should evolve with weights
#   second pooling layer of 64  - pool all 16*4 images
#   3 hidden layers of 64       - 3 weighted layers of 64 neurons each with corresponding weights in the weights matrix
#   output layer of 131         - 131 output neurons can be stored as list[] * 131; each output neuron should receive input from each neuron in the final hidden layer

class CNN:
    def __init__(self, layers, epochs):
        self.classes = ["Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1", "Apple Red 2", "Apple Red 3", "Apple Red Delicious", "Apple Red Yellow 1", "Apple Red Yellow 2", "Apricot", "Avocado", "Avocado ripe", "Banana", "Banana Lady Finger", "Banana Red", "Beetroot", "Blueberry", "Cactus fruit", "Cantaloupe 1", "Cantaloupe 2", "Carambula", "Cauliflower", "Cherry 1", "Cherry 2", "Cherry Rainier", "Cherry Wax Black", "Cherry Wax Red", "Cherry Wax Yellow", "Chestnut", "Clementine", "Cocos", "Corn", "Corn Husk", "Cucumber Ripe", "Cucumber Ripe 2", "Dates", "Eggplant", "Fig", "Ginger Root", "Granadilla", "Grape Blue", "Grape Pink", "Grape White", "Grape White 2", "Grape White 3", "Grape White 4", "Grapefruit Pink", "Grapefruit White", "Guava", "Hazelnut", "Huckleberry", "Kaki", "Kiwi", "Kohlrabi", "Kumquats", "Lemon", "Lemon Meyer", "Limes", "Lychee", "Mandarine", "Mango", "Mango Red", "Mangostan", "Maracuja", "Melon Piel de Sapo", "Mulberry", "Nectarine", "Nectarine Flat", "Nut Forest", "Nut Pecan", "Onion Red", "Onion Red Peeled", "Onion White", "Orange", "Papaya", "Passion Fruit", "Peach", "Peach 2", "Peach Flat", "Pear", "Pear 2", "Pear Abate", "Pear Forelle", "Pear Kaiser", "Pear Monster", "Pear Red", "Pear Stone", "Pear Williams", "Pepino", "Pepper Green", "Pepper Orange", "Pepper Red", "Pepper Yellow", "Physalis", "Physalis with Husk", "Pineapple", "Pineapple Mini", "Pitahaya Red", "Plum", "Plum 2", "Plum 3", "Pomegranate", "Pomelo Sweetie", "Potato Red", "Potato Red Washed", "Potato Sweet", "Potato White", "Quince", "Rambutan", "Raspberry", "Redcurrant", "Salak", "Strawberry", "Strawberry Wedge", "Tamarillo", "Tangelo", "Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato Cherry Red", "Tomato Heart", "Tomato Maroon", "Tomato not Ripened", "Tomato Yellow", "Walnut", "Watermelon"]
        self.neurons = 48
        self.layers = layers
        self.network = np.zeros((self.layers,16))
        train_count, test_count, path = self.getRepo()
        train_data, test_data = self.getData(path)
        self.fit(train_count, train_data, epochs)

    def getRepo(self):
        path = os.getcwd() + '/Fruit'
        if os.path.exists(path):
            # do nothing
            print("Local repo already exists.")
        else:
            cloned = pygit2.clone_repository("https://github.com/ReeseReynolds/ML-Testing", path, bare=False,
                                             repository=None, remote=None, checkout_branch=None, callbacks=None)

        # temp dir for testing
        if not os.path.exists(os.getcwd() + '/out'):
            os.mkdir(os.getcwd() + '/out')
            os.chmod(os.getcwd() + '/out', 0o777)

        # calc file size of train folder
        train_count = sum(len(files) for _, _, files in os.walk(path + '/Train'))
        print("Training set size: ", train_count)

        # calc file size of test folder
        test_count = sum(len(files) for _, _, files in os.walk(path + '/Test'))
        print("Test set size: ", test_count)
        return train_count, test_count, path

    def getData(self, path):
        # gather all class labels and file names
        train_set = []
        test_set = []
        for root, dirs, files in os.walk(path):
            if 'Train' in root:
                for file in files:
                    p = root.replace(path, '')
                    p = p.replace('Train', '')
                    p = p.replace(str(file), '')
                    label = p.replace('\\', '')
                    # print([root + "\\" + file, self.classes.index(label), file])
                    train_set.append([root + "\\" + file, self.classes.index(label), file])

            if 'Test' in root:
                for file in files:
                    p = root.replace(path, '')
                    p = p.replace('Test', '')
                    p = p.replace(str(file), '')
                    label = p.replace('\\', '')
                    test_set.append([root + "\\" + file, self.classes.index(label), file])

        return train_set, test_set

    def genSlices(self, raw_im):
        slices = []
        # slices[0-3] generate and store 4 50x50x3 images from all quadrants of raw_im (RGB so there are 3 color channels)
        # slice image into four quadrants
        slices.append(raw_im[:50,:50])
        slices.append(raw_im[:50, 50:])
        slices.append(raw_im[50:,:50])
        slices.append(raw_im[50:,50:])

        # for p in range(4):
            # o_name = "out\\slice_" + str(p) + ".jpg"
            # imageio.imwrite(o_name, slices[p])

        return slices

    def ReLU(self, x):
        # TODO simple max ReLU function to be called after each layer
        return np.maximum(0, x)

    def conv(self, rgb_ims, filters, size):
        # pool will be filled with groups of four:
        # each group consists of one base image with four unique filters applied
        pool = []
        dt = np.dtype('uint8')
        for im in rgb_ims:
            for f in filters:
                filtered_im = []
                for row in im:
                    row_slice = [] * len(row)
                    for i in range(len(row) - 2):
                        row_slice[i:i + 3] = np.matmul(f, row[i:i + 3])
                    filtered_im.append(row_slice)

                filtered_im = np.array(filtered_im, dtype=dt)
                pool.append(np.reshape(filtered_im, (size, size, 3)))

        # for i in range(len(pool)):
            # o_name = "out\\conv_" + str(i) + ".jpg"
            # imageio.imwrite(o_name, pool[i])
        return pool

    def convReLU(self, rgb_ims):
        imgs_final = []
        for i in range(0, len(rgb_ims), 4):
            group = rgb_ims[i:i+4]
            for img in group:
                t_img = []
                for row in img:
                    t_row = []
                    for pix in row:
                        t_pix = [self.ReLU(pix[i]) for i in range(len(pix))]
                        t_row.append(t_pix)
                    t_img.append(t_row)
                imgs_final.append(t_img)
        return imgs_final

    def pool(self, rgb_ims, size, n):
        # max pooling algorithm
        pool = []
        dt = np.dtype('uint8')
        for i in range(0, len(rgb_ims), 4):
            imgs_final = []
            group = rgb_ims[i:i + 4]
            for img in group:
                t_img = []
                for row in img:
                    t_row = []
                    for pix in row:
                        t_pix = [self.ReLU(pix[i]) for i in range(len(pix))]
                        t_row.append(t_pix)
                    t_img.append(t_row)
                imgs_final.append(t_img)
            # apply max pooling
            temp_pool = []
            for im in imgs_final:
                fin = [] * int((size / 2))
                for j in range(int(size/n)):
                    fin_row = []
                    for k in range(int(size/n)):
                        image = np.array(im)
                        block = image[j*n:(j*n)+n,k*n:(k*n)+n]

                        block = np.array(block, dtype=dt)
                        block = np.reshape(block, (3, n*n))
                        pix = [max(block[q]) for q in range(3)]
                        fin_row.append(pix)
                    fin.append(fin_row)
                temp_pool.append(fin)

            # pool group together into a single image
            # print(np.shape(temp_pool[0]))
            rs = np.add(temp_pool[0], temp_pool[1])
            rs2 = np.add(temp_pool[2], temp_pool[3])
            fin_img = np.add(rs, rs2)

            fin_img = np.array(fin_img, dtype=dt)
            pix_fin = np.reshape(fin_img, (int(size/n), int(size/n), 3))
            # o_name = "out/pool_test.jpg"
            # imageio.imwrite(o_name, pix_fin)

            pool.append(pix_fin)

        return pool

    def train(self, slices, weights, filters):
        # create all layers, some should require extra parameters to be implemented
        # TODO implement backpropagation
        conv_out = self.conv(slices, filters[:4], 50)
        ReLU_out = self.convReLU(conv_out)
        pool1_out = self.pool(ReLU_out, 50, 2)

        conv_out = self.conv(pool1_out, filters[4:], 25)
        ReLU_out = self.convReLU(conv_out)
        pool2_out = self.pool(ReLU_out, 25, 5)

        # dnn strongly connected layers

        return weights

    def fit(self, file_count, img_set, epochs):
        # loads and trains with 10k images at a time until all images have been iterated through
        # generate weights randomly, 16 weights for each layer
        weights = np.random.random((self.layers, 256))
        filters = [np.random.randint(-10, 10, (3, 3)) for _ in range(8)]
        for e in range(epochs):
            print("Beginning epoch ", e)
            start = time.time()
            for i in range(file_count):
                # get image
                fname = img_set[i][0]
                im = Image.open(fname, 'r')
                pix = list(im.getdata())
                im.close()

                dt = np.dtype('uint8')
                pix_arr = np.array(pix, dtype=dt)
                pix_fin = np.reshape(pix_arr, (100, 100, 3))

                # o_name = "out/out0.jpg"
                # imageio.imwrite(o_name, pix_fin)
                im_slices = self.genSlices(pix_fin)
                weights = self.train(im_slices, weights, filters)


                if (i+1) % 11282 == 0 and (i+1) != 0:
                    print("Group of 10k images processed")
                    print(time.time() - start, " seconds for i = ", i)
                    start = time.time()

st = time.time()
p = CNN(3,1)
print("Total execution time: ", time.time()-st)