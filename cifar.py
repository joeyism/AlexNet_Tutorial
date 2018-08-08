import pickle
import os


def __extract_file__(fname):
    with open(fname, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def __unflatten_image__(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img

  
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
        self.batches = []
        self.__batch_num__ = 0
        for i in range(math.ceil(len(self.__res__)/batch_size)):
            self.batches.append(self.__res__[i*batch_size:(i+1)*batch_size])

    def batch(self, num):
        return self.batches[num]

    def next_batch(self):
        if self.__batch_num__ <= len(self.batches):
            res = self.batches[self.__batch_num__]
            self.__batch_num__ = self.__batch_num__ + 1
        else:
            res = []

        return res

    def reset_batch(self):
        self.__batch_num__ = 0