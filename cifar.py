from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pickle
import math
import helper
import os
import gc


def __extract_file__(fname):
    with open(fname, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def __unflatten_image__(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img/255


def __extract_reshape_file__(fname):
    res = []
    d = __extract_file__(fname)
    images = d[b"data"]
    labels = d[b"labels"]
    for image, label in zip(images, labels):
        res.append((__unflatten_image__(image), label))
    return res


def get_images_from(dir):
    files = [f for f in os.listdir(dir) if f.startswith("data_batch")]
    res = []
    for f in files:
        res = res + __extract_reshape_file__(os.path.join(dir, f))
    return res


class Cifar(object):

    def __init__(self, dir="data/cifar-10-batches-py/", batch_size=1):
        self.__res__ = get_images_from(dir)
        self.batch_size = batch_size
        self.no_of_batches = math.ceil(len(self.__res__)/batch_size)
        self.batches = []
        self.__batch_num__ = 0
        for i in range(self.no_of_batches):
            self.batches.append(self.__res__[i*batch_size:(i+1)*batch_size])
        self.test_set = __extract_reshape_file__(os.path.join(dir, "test_batch"))

    def batch(self, num):
        return self.batches[num]

    def resized_batch(self, num, at="data/resized_batches"):
        with open(at+"/batch{}.pickle".format(str(num)), "wb") as f:
            b = pickle.load(f)
        return b

    def next_batch(self):
        if self.__batch_num__ <= len(self.batches):
            res = self.batches[self.__batch_num__]
            self.__batch_num__ = self.__batch_num__ + 1
        else:
            res = []

        return res

    def reset_batch(self):
        self.__batch_num__ = 0

    def create_resized_test_set(self, new_size=(224, 224), dim=1000):
        self.test_set = helper.transform_to_input_output_and_pad(self.test_set, new_size=new_size, dim=dim)

    def create_resized_batches(self, new_size=(224, 224), dim=1000):
        self.resized_batches = [helper.transform_to_input_output_and_pad(batch, new_size=new_size, dim=dim) for batch in tqdm(self.batches, desc="Reshaping")]
        self.create_resized_test_set(new_size, dim)

    def save_resized_batches(self, to="data/resized_batches", new_size=(224, 224), dim=10):

        def write(i, batch):
            this_batch = helper.reshape_batch(batch, new_size, dim)
            with open(to+"/batch{}.pickle".format(str(i)), "wb") as f:
                pickle.dump(this_batch, f)
            del this_batch
            gc.collect()

        if not os.path.exists(to):
            os.makedirs(to)

        for i, batch in tqdm(list(enumerate(self.batches)), desc="reshaping data"):
            write(i, batch)
