# import imageio
import numpy as np
import time
from PIL import Image
import pygit2
import os
import math
import random

# topology:
#   input layer of 1              - one rgb image
#   first conv layer of 16        - four convolutions for features
#   pooling layer of 16           - max pooling layer downscales 50x50 to 25x25
#   second conv layer of 64       - four new filters to apply to all previous 16 images which should evolve with weights
#   second pooling layer of 64    - max pooling layer downscales 25x25 to 5x5
#   3 hidden layers of 16         - 3 strongly connected layers of 16 neurons
#   output layer of 131           - 131 output neurons

class CNN:
    # Creates a CNN that operates on a certain number of images from the train and test datasets.
    def __init__(self, layers, epochs, train_sample_size=None, test_sample_size=None):
        self.classes = ['Apple Braeburn', 'Apricot', 'Avocado', 'Banana', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Carambula', 'Cauliflower', 'Cherry 1', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Cucumber Ripe', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grapefruit Pink', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nut Forest', 'Onion Red', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepino', 'Pepper Green', 'Physalis', 'Pineapple', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Walnut', 'Watermelon']
        self.neurons = 16
        self.layers = layers
        self.filters = [np.random.randint(-10, 10, (3, 3)) for _ in range(8)]
        self.network = np.zeros((self.layers, 16))
        self.error = np.zeros((self.layers-1, 16))
        self.weights = np.random.uniform(-0.3, 0.3, (self.layers-1, 256))
        self.out_weights = np.random.uniform(-0.3, 0.3, 1056)
        self.out_error = np.zeros(66)
        path = self.getRepo()

        if train_sample_size is None:
            train_sample_size = -1
        if test_sample_size is None:
            test_sample_size = -1

        train_data, test_data = self.getData(path, train_sample_size, test_sample_size)
        self.fit(len(train_data), train_data, epochs)
        self.predict(len(test_data), test_data)

    # Downloads dataset to local working directory.
    # Returns local file path of data set.
    def getRepo(self):
        path = os.getcwd() + '/Fruit'
        if os.path.exists(path):
            # do nothing
            print("Local repo exists.")
        else:
            cloned = pygit2.clone_repository("https://github.com/ReeseReynolds/ML-Testing", path, bare=False,
                                             repository=None, remote=None, checkout_branch=None, callbacks=None)

        # temp dir for testing
        if not os.path.exists(os.getcwd() + '/out'):
            os.mkdir(os.getcwd() + '/out')
            os.chmod(os.getcwd() + '/out', 0o777)

        return path

    # Stores given numbers of images from the training/test data sets
    # Returns lists containing training and test samples.
    def getData(self, path, size_train, size_test):
        # gather all class labels and file names
        train_set = []
        test_set = []
        for root, dirs, files in os.walk(path):
            if 'Train' in root:
                i = 0
                for file in files:
                    if i == size_train:
                        break
                    i += 1
                    p = root.replace(path, '')
                    p = p.replace('Train', '')
                    p = p.replace(str(file), '')
                    label = p.replace('\\', '')
                    train_set.append([root + "\\" + file, self.classes.index(label), file])
            if 'Test' in root:
                i = 0
                for file in files:
                    if i == size_test:
                        break
                    i += 1
                    p = root.replace(path, '')
                    p = p.replace('Test', '')
                    p = p.replace(str(file), '')
                    label = p.replace('\\', '')
                    test_set.append([root + "\\" + file, self.classes.index(label), file])

        return train_set, test_set

    # Slices given image into four 1/4 size quadrants.
    # Returns list of 4 quadrants.
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

    # Simple ReLU function
    def ReLU(self, x):
        return np.maximum(0, x)

    # Simple sigmoid function bound to the range (0,1)
    def sigmoid(self, x):
        x = 1 / (1 + np.exp(-x))
        x = np.minimum(x, 0.9999)
        x = np.maximum(x, 0.0001)
        return x

    # Driver to apply ReLU to all pixels in the input images
    # Returns same list with ReLU applied to all members.
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

    # Applies four convolution filters to each image in input list and maintains image size by using stride of 1.
    # Returns list of len(rgb_ims) * 4 convoluted images.
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

    # Pools filtered versions of the same image into a single image and applies max pooling algorithm to the result.
    # Returns list of max pooled images
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

            for fin_img in temp_pool:
                fin_img = np.array(fin_img, dtype=dt)
                pix_fin = np.reshape(fin_img, (int(size/n), int(size/n), 3))
                pool.append(pix_fin)

        return pool

    # Runs input images through NN
    # Returns calculated output predictions
    def dnn(self, rgb_ims):
        for i in range(len(self.network)):
            for j in range(16):
                pred = 0
                # input layer
                if i == 0:
                    for im in rgb_ims[(j*4):(j*4)+4]:
                        im = np.ndarray.flatten(im)
                        pred += sum([x for x in im])/len(im)

                    self.network[i][j] = self.sigmoid(math.sqrt(pred))
                # hidden layers
                else:
                    summ = 0
                    for k in range(16):
                        summ += self.network[i-1][k] * self.weights[i-1][j*16+k]
                    self.network[i][j] = self.sigmoid(summ)
        out = []
        for i in range(66):
            summ = 0
            for j in range(16):
                # print(self.network[-1][j])
                summ += self.network[-1][j] * self.out_weights[i*16+j]
            out.append(self.sigmoid(summ))
        print("Prediction: ", ["%.2f" % o for o in out])
        return out

    # derivative of sigmoid function used for calculating error
    def derivative(self, x):
        return x * (1.0 - x)

    # Backpropagation of errors and updating weights
    def backpropagate(self, calc, actual):
        lr = 0.03

        # recalc weights for output layer
        for i in range(len(calc)):
            self.out_error[i] = calc[i] - actual[i]
            for j in range(16):
                self.out_weights[i*16+j] = self.out_weights[i*16+j] - lr * self.out_error[i] * self.network[-1][j]

        # recalc weights for hidden layers
        for i in reversed(range(len(self.weights))):
            for j in range(16):
                summ = 0
                if i == len(self.weights)-1:
                    for k in range(16):
                        for l in range(66):
                            summ += self.weights[i][j*16+k] * self.out_error[l] * self.derivative(self.network[i][j])
                    self.error[i][j] = summ
                else:
                    for k in range(16):
                        for q in range(16):
                            summ += self.weights[i][j*16 + k] * self.error[i+1][q] * self.derivative(self.network[i][j])
                    self.error[i][j] = summ
                self.weights[i][j*16+k] = self.weights[i][j*16+k] - lr * self.error[i][j] * self.network[i][j]

    # Predicts most likely output class for each image in input set
    def predict(self, file_count, img_set):
        print("Test File count: ", file_count)
        for i in range(file_count):
            stt = time.time()
            fname = img_set[i][0]
            actual = img_set[i][1]
            im = Image.open(fname, 'r')
            pix = list(im.getdata())
            im.close()

            dt = np.dtype('uint8')
            pix_arr = np.array(pix, dtype=dt)
            pix_fin = np.reshape(pix_arr, (100, 100, 3))

            im_slices = self.genSlices(pix_fin)
            prediction = self.test(im_slices, self.filters)
            print("Time for prediction ", i, ": ", time.time() - stt)
            print("Prediction: ", self.classes[prediction])
            print("Actual: ", self.classes[actual])

    # Processes input images through two convolution and pooling layers, and a NN.
    def train(self, slices, filters, actual):
        # create all layers, some should require extra parameters to be implemented
        conv_out = self.conv(slices, filters[:4], 50)
        ReLU_out = self.convReLU(conv_out)
        pool1_out = self.pool(ReLU_out, 50, 2)

        conv_out = self.conv(pool1_out, filters[4:], 25)
        ReLU_out = self.convReLU(conv_out)
        pool2_out = self.pool(ReLU_out, 25, 5)

        # dnn strongly connected layers
        # self.weights, compare actual with predicted
        out = self.dnn(pool2_out)
        rs = max(out)
        print("Train: ", self.classes[out.index(rs)])
        print("Actual: ", self.classes[actual])

        act = np.zeros(66)
        act[actual] = 1
        self.backpropagate(out, act)

    # Run test image through CNN and compute predicted outputs.
    # Return most likely output index.
    def test(self, slices, filters):
        # create all layers, some should require extra parameters to be implemented
        conv_out = self.conv(slices, filters[:4], 50)
        ReLU_out = self.convReLU(conv_out)
        pool1_out = self.pool(ReLU_out, 50, 2)

        conv_out = self.conv(pool1_out, filters[4:], 25)
        ReLU_out = self.convReLU(conv_out)
        pool2_out = self.pool(ReLU_out, 25, 5)

        out = self.dnn(pool2_out)
        rs = max(out)
        return out.index(rs)

    # Generates convolution filters and trains model over all images for every epoch.
    def fit(self, file_count, img_set, epochs):
        print("Train File count: ", len(img_set))
        # randomize the order which images will be picked for training
        im_idx = list(range(file_count))
        random.shuffle(im_idx)

        for e in range(epochs):
            print("Beginning epoch ", e)
            start = time.time()
            for i in range(file_count):
                fname = img_set[im_idx[i]][0]
                actual = img_set[im_idx[i]][1]
                im = Image.open(fname, 'r')
                pix = list(im.getdata())
                im.close()

                dt = np.dtype('uint8')
                pix_arr = np.array(pix, dtype=dt)
                pix_fin = np.reshape(pix_arr, (100, 100, 3))

                im_slices = self.genSlices(pix_fin)
                self.train(im_slices, self.filters, actual)
            print("Epoch ", e, " finished: ", start - time.time())

st = time.time()
p = CNN(3,1,8,5)
print("Total execution time: ", time.time()-st)