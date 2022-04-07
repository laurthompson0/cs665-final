"""
Functions that take in an image and return an augmented version of that image
(Images are expected to be in PIL format, but you could easily modify
the `augment_positives` script to make this take np arrays or any other 
image loading framework)
"""


def flip(image):
    # TODO
    return image


def mirror(image):
    # TODO
    return image


def crop(image):
    # TODO
    return image


def noise(image):
    # TODO
    return image


AUGMENTATION_FUNCTIONS = {
    "flip": flip,
    "mirror": mirror,
    "crop": crop,
    "noise": noise,
}
