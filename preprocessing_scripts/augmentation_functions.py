"""
Functions that take in an image and return an augmented version of that image
(Images are expected to be in PIL format, but you could easily modify
the `augment_positives` script to make this take np arrays or any other 
image loading framework)
"""
from PIL import Image
from random import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.util import random_noise

def brighten(image):
    data = img_to_array(image) / 255
    image = data * 1.5
    return image

def darken(image):
    data = img_to_array(image) / 255
    image = data * 0.5
    return image


def mirror(image):
    data = img_to_array(image)
    mirror = np.fliplr(data)
    image = np.uint8(mirror)
    return image


def crop(image, crop_percent = 0.7, rand_override=None):
    MINSIZE = 224
    width, height = image.size

    x = MINSIZE if round(width * crop_percent) < MINSIZE else round(width * crop_percent)
    y = MINSIZE if round(height * crop_percent) < MINSIZE else round(height * crop_percent)

    rand = rand_override if rand_override else random()
    if rand <= .2:
        # crop from top left
        image = image.crop((0, 0, x, y))
    if .2 < rand and rand <= .4:
        # crop from bottom left
        image = image.crop((0, height-y, x, height))
    if .4 < rand and rand <= .6:
        # crop from top right
        image = image.crop((width-x, 0, width, y))
    if .6 < rand and rand <= .8:
        # crop from bottom right
        image = image.crop((width-x, height-y, width, height))
    if .8 < rand:
        # crop from middle
        image = image.crop(((width-x)/2, (height-y)/2, (width+x)/2, (height+y)/2))

    return image


def noise(image):
    data = img_to_array(image) / 255
    noisy = random_noise(data)
    return noisy


AUGMENTATION_FUNCTIONS = {
    "brighten": brighten,
    "darken": darken,
    "mirror": mirror,
    "crop": crop,
    "noise": noise,
}


def main():
    """
    Command line test function. Allows testing of individual functions on singular images.

    Usage: ./augmentation_functions.py -f <func_name> i <img_path>
    func_name: from AUGMENTATION_FUNCTIONS
    img_path: system path to image
    """
    if len(sys.argv) > 2:
        try:
            funcIdx = sys.argv.index("-f")
        except ValueError:
            funcIdx = None
        try:
            imgIdx = sys.argv.index("-i")
        except ValueError:
            imgIdx = None
        
        if funcIdx and funcIdx != len(sys.argv)-1 and funcIdx != imgIdx: 
            func = sys.argv[funcIdx+1]
        else:
            sys.exit('Function required. -f <function_name>')
        if imgIdx and imgIdx != len(sys.argv)-1 and funcIdx != imgIdx: 
            imgIdx = imgIdx+1
        else:
            sys.exit('Image file required. -i <path>')
        
        img = Image.open(sys.argv[imgIdx])
        img_edited = AUGMENTATION_FUNCTIONS[func](img)
        img_edited.show()

if __name__ == "__main__": import sys; main()
