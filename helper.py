import numpy as np
from PIL import Image
import skimage.transform


def one_hot(num, dim=1000):
    vec = np.zeros(dim)
    vec[num] = 1
    return vec.tolist()


def transform_to_input_output(input_output, dim=1000):
    input_vals = []
    output_vals = []
    for input_val, output_val in input_output:
        input_vals.append(input_val)
        output_vals.append(output_val)

    return np.array(input_vals), np.array(
        [one_hot(out, dim=dim)
         for out in output_vals],
        dtype="uint8")


def reshape(image, new_size):
    return skimage.transform.resize(image, new_size, mode="constant")


def transform_to_input_output_and_pad(input_output, new_size=(224, 224), dim=1000):
    inp, out = transform_to_input_output(input_output, dim=dim)
    return np.array([reshape(i, new_size) for i in inp]), out


def reshape_batch(batch, new_size, dim=10):
    input_batch=[]
    output_batch=[]
    for image, out in batch:
        new_image = reshape(image, new_size)
        input_batch.append(new_image)
        output_batch.append(one_hot(out, dim=dim))

    return input_batch, output_batch
        
