import sys
import pyvips
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def fetch(region, patch_size, x, y):
    return region.fetch(patch_size * x, patch_size * y, patch_size, patch_size)
image = [i for i in range(8)]
__format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
}

image[0] = pyvips.Image.new_from_file(sys.argv[1],level=0)
image[1] = image[0].rot90()
image[2] = image[0].rot180()
image[3] = image[0].rot270()

image[4] = image[0].fliphor()
image[5] = image[4].rot90()
image[6] = image[4].rot180()
image[7] = image[4].rot270()

reg = [pyvips.Region.new(x) for x in image]

patch_size = 224
n_across = image[0].width // patch_size
n_down = image[0].height // patch_size
x_max = n_across - 1
y_max = n_down - 1

n_patches = 0
for y in tqdm(range(0, n_down)):
    print("row {} ...".format(y))
    for x in tqdm(range(0, n_across)):
        patch0 = fetch(reg[0], patch_size, x, y)
        interm = np.ndarray(buffer=patch0, dtype=np.uint8, shape=[patch_size,patch_size,4])
        patch1 = fetch(reg[1], patch_size, y_max - y, x)
        interm = np.ndarray(buffer=patch1, dtype=np.uint8, shape=[patch_size,patch_size,4])
        patch2 = fetch(reg[2], patch_size, x_max - x, y_max - y)
        interm = np.ndarray(buffer=patch2, dtype=np.uint8, shape=[patch_size,patch_size,4])
        patch3 = fetch(reg[3], patch_size, y, x_max - x)
        interm = np.ndarray(buffer=patch3, dtype=np.uint8, shape=[patch_size,patch_size,4])

        patch4 = fetch(reg[4], patch_size, x_max - x, y)
        interm = np.ndarray(buffer=patch4, dtype=np.uint8, shape=[patch_size,patch_size,4])
        patch5 = fetch(reg[5], patch_size, y_max - y, x_max - x)
        interm = np.ndarray(buffer=patch5, dtype=np.uint8, shape=[patch_size,patch_size,4])
        patch6 = fetch(reg[6], patch_size, x, y_max - y)
        interm = np.ndarray(buffer=patch6, dtype=np.uint8, shape=[patch_size,patch_size,4])
        patch7 = fetch(reg[7], patch_size, y, x)
        interm = np.ndarray(buffer=patch7, dtype=np.uint8, shape=[patch_size,patch_size,4])

        n_patches += 8

print("{} patches generated".format(n_patches))
