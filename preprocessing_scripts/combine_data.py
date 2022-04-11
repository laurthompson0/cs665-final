"""
This script is intended to combine the data
for negative and positive cases from the two folders
`positive_data` and `negative_data` in a random order
and write a ground truth document, `image_labels.txt` for the images.


Assumptions:
    - parent directory for combined images is named "data"
    - Negative images are given first
"""

import csv
from shutil import copy, rmtree
from os.path import join
from os import mkdir, listdir
from random import shuffle
from tkinter.ttk import LabeledScale
from typing import Callable
from tensorflow.keras.preprocessing.image import load_img, save_img


def copy_rename_augment(
    old_dir: str,
    new_dir: str,
    filename: str,
    file_num: int,
    dir_num: int,
    aug_functions,
):
    """Copy from one path to another and rename according to file_num"""
    old_path = join(old_dir, filename)
    new_fn = f"{str(file_num).zfill(4)}.{filename.split('.')[1]}"
    new_path = join(new_dir, new_fn)
    image = load_img(old_path, target_size=(224, 224))  # Resize
    save_img(new_path, image)

    if dir_num == 1:
        # Augment positive images
        image = load_img(new_path, target_size=(224, 224))  # PIL
        for aug_type, aug_function in aug_functions.items():
            file_num += 1
            aug_image = aug_function(image)  # Augment
            aug_name = f"{str(file_num).zfill(4)}.{filename.split('.')[1]}"
            save_img(join(new_dir, aug_name), aug_image)


def copy_rename(
    old_dir: str,
    new_dir: str,
    filename: str,
    file_num: int,
):
    """Copy from one path to another and rename according to file_num"""
    old_path = join(old_dir, filename)
    new_fn = str(file_num).zfill(4) + "." + filename.split(".")[1]
    new_path = join(new_dir, new_fn)
    copy(old_path, new_path)


def shuffle_imgs(labels, tempdir, imgdir, offset, shuffle_labels):
    if shuffle_labels:
        shuffle(labels)
    for ind, label in enumerate(labels):
        try:
            copy_rename(
                tempdir, join(imgdir, "images"), label[0] + ".jpg", ind + offset + 1
            )  # copy files in shuffled order
        except FileNotFoundError:
            copy_rename(
                tempdir, join(imgdir, "images"), label[0] + ".png", ind + offset + 1
            )  # just some sketchy error handling to accound for multiple data types

    labels = [
        [str(ind + offset + 1).zfill(4), label[1]] for ind, label in enumerate(labels)
    ]  # rename labels based on their new index

    return labels


def combine(dirs: list, augmentation_functions, split=0.8, shuffle_labels=True):
    """Combine positive and negative image dirs into one, shuffle, and generate label csv"""

    # Initialize file counter and label list
    file_num = 1
    labels_test = []
    labels_train = []
    num_funcs = len(augmentation_functions.items())
    tempdir_test = "data/images_temp_test"
    tempdir_train = "data/images_temp_train"
    imgdir_test = "data/test"
    imgdir_train = "data/train"

    # Clear image folders if exists
    rmtree("data", ignore_errors=True)
    mkdir("data")
    mkdir(imgdir_test)
    mkdir(imgdir_train)
    mkdir(tempdir_test)  # for copying data
    mkdir(tempdir_train)
    mkdir(join(imgdir_train, "images"))
    mkdir(join(imgdir_test, "images"))

    # Iterate over each dir and copy
    for dir_num, dir in enumerate(dirs):
        file_num_dir = 1
        filenames = listdir(dir)
        num_files = len(filenames)

        for filename in filenames:
            if (file_num_dir / num_files) < split:
                copy_rename_augment(
                    dir,
                    tempdir_train,
                    filename,
                    file_num,
                    dir_num,
                    augmentation_functions,
                )
                labels_train.append(
                    [str(file_num).zfill(4), dir_num]
                )  # NEGATIVES MUST COME FIRST FOR THIS TO WORK
                # Add labels for augmented images
                if dir_num == 1:
                    for i in range(num_funcs):
                        file_num += 1
                        labels_train.append([str(file_num).zfill(4), dir_num])
            else:
                copy_rename_augment(
                    dir,
                    tempdir_test,
                    filename,
                    file_num,
                    dir_num,
                    augmentation_functions,
                )
                labels_test.append(
                    [str(file_num).zfill(4), dir_num]
                )  # NEGATIVES MUST COME FIRST FOR THIS TO WORK
                if dir_num == 1:
                    # Add labels for augmented images
                    for i in range(num_funcs):
                        file_num += 1
                        labels_test.append([str(file_num).zfill(4), dir_num])
            file_num += 1
            file_num_dir += 1

    # Shuffle and write label data
    label_list = [
        (labels_train, tempdir_train, imgdir_train),
        (labels_test, tempdir_test, imgdir_test),
    ]
    for i, label_info in enumerate(label_list):
        offset = i * len(label_list[i - 1][0])
        new_labels = shuffle_imgs(
            label_info[0], label_info[1], label_info[2], offset, shuffle_labels
        )
        with open(join(label_info[2], "labels.csv"), "w") as label_file:
            filewriter = csv.writer(label_file)
            filewriter.writerows(new_labels)
        # delete tempdir
        rmtree(label_info[1], ignore_errors=True)
