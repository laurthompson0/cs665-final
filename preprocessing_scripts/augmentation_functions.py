"""
Functions that take in an image and return an augmented version of that image
(Images are expected to be in PIL format, but you could easily modify
the `augment_positives` script to make this take np arrays or any other 
image loading framework)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.util import random_noise

def flip(image):
    # TODO
    return image


def mirror(image):
    data = img_to_array(image)
    mirror = np.fliplr(data)
    image = np.uint8(mirror)
    return image


def crop(image):
    # TODO
    return image


def noise(image):
    data = img_to_array(image) / 255
    noisy = random_noise(data)
    return noisy


AUGMENTATION_FUNCTIONS = {
    "flip": flip,
    "mirror": mirror,
    "crop": crop,
    "noise": noise,
}
