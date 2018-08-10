import numpy as np


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