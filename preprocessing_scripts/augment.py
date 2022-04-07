"""
This script is intended to combine the data
for negative and positive cases from the two folders
`positive_data` and `negative_data` in a random order
and write a ground truth document, `image_labels.txt` for the images.


Assumptions:
    - There is an existing folder called `positive_data` containing images
        of social distancing with extensions either '.jpg' or '.png'
    - These images will be augmented in some way and then renamed accordingly
"""

from shutil import rmtree
from os.path import join
from os import mkdir, listdir
from typing import Callable
from tensorflow.keras.preprocessing.image import load_img, save_img


def augment(
    filenames: list[str],
    old_dir: str,
    new_dir: str,
    augmentation_type: str,
    augmentation_func: Callable,
):
    """
    Iterates over the provided filenames, runs the specified augmentation functions
    and saves the new image with a name corresponding to the type of augmentation
    """
    for filename in filenames:
        image = load_img(join(old_dir, filename), target_size=(224, 224))  # PIL
        aug_image = augmentation_func(image)  # Augment
        aug_name = f"{filename[:-4]}_{augmentation_type}{filename[-4:]}"
        save_img(join(new_dir, aug_name), aug_image)


def augment_data(old_dir: str, new_dir: str, augmentation_funcs: dict):
    """
    Given an old directory and new one, augment images and save them to new path
    Note, also saves original images to the new path
    """
    # Clear image folder if it exists
    rmtree(new_dir, ignore_errors=True)
    mkdir(new_dir)

    filenames = listdir(old_dir)

    # Run each augmentation function on all files
    for augmentation_type, augmentation_func in augmentation_funcs.items():
        augment(filenames, old_dir, new_dir, augmentation_type, augmentation_func)

    # Save existing images to new directory
    for filename in filenames:
        save_img(
            join(new_dir, filename),
            load_img(join(old_dir, filename), target_size=(224, 224)),
        )
