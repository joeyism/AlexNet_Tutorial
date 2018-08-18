import numpy as np
from PIL import Image


def __one_hot__(num, dim=1000):
    vec = np.zeros(dim)
    vec[num] = 1
    return vec


def transform_to_input_output(input_output, dim=1000):
    input_vals = []
    output_vals = []
    for input_val, output_val in input_output:
        input_vals.append(input_val)
        output_vals.append(output_val)

    return np.array(input_vals), np.array(
        [__one_hot__(out, dim=dim)
         for out in output_vals],
        dtype="uint8")


def add_padding(image, new_size):
    old_im = Image.fromarray(image, "RGB")
    old_size = old_im.size

    new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
    new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),
                              int((new_size[1]-old_size[1])/2)))

    return np.array(new_im)


def transform_to_input_output_and_pad(input_output, new_size=(227, 227), dim=1000):
    inp, out = transform_to_input_output(input_output, dim=dim)
    return np.array([add_padding(i, new_size) for i in inp]), out

