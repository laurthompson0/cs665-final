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


def copy_rename(
    old_dir: str,
    new_dir: str,
    filename: str,
    file_num: int,
):
    """Copy from one path to another and rename according to file_num"""
    old_path = join(old_dir, filename)
    new_fn = f"{str(file_num).zfill(4)}.{filename.split('.')[1]}"
    new_path = join(new_dir, new_fn)
    image = load_img(old_path, target_size=(224, 224))  # Resize
    save_img(new_path, image)


def copy_rename_augment(
    old_dir: str,
    new_dir: str,
    filename: str,
    file_num: int,
    aug_functions,
):
    """Copy from one path to another and rename according to file_num"""
    old_path = join(old_dir, filename)
    new_fn = f"{str(file_num).zfill(4)}.{filename.split('.')[1]}"
    new_path = join(new_dir, new_fn)
    image = load_img(old_path, target_size=(224, 224))  # Resize
    aug_func = list(aug_functions.values())[0]  # Get just first function
    aug_img = aug_func(image)
    save_img(new_path, aug_img)


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
    labels_val = []
    num_funcs = len(augmentation_functions.items())
    tempdir_test = "data/images_temp_test"
    tempdir_train = "data/images_temp_train"
    tempdir_val = "data/images_temp_val"
    imgdir_test = "data/test"
    imgdir_train = "data/train"
    imgdir_val = "data/val"

    # Clear image folders if exists
    rmtree("data", ignore_errors=True)
    mkdir("data")
    mkdir(imgdir_test)
    mkdir(imgdir_train)
    mkdir(imgdir_val)
    mkdir(tempdir_test)  # for copying data
    mkdir(tempdir_train)
    mkdir(tempdir_val)
    mkdir(join(imgdir_train, "images"))
    mkdir(join(imgdir_test, "images"))
    mkdir(join(imgdir_val, "val"))

    # Iterate over each dir and copy
    for dir_num, dir in enumerate(dirs):
        file_num_dir = 1
        filenames = listdir(dir)
        num_files = len(filenames)
        num_test = num_files / (2 - split)
        train_split = (num_files - num_test) / num_files
        test_split = train_split + (1 - train_split) / 2

        # get num train to determine augmentation
        if dir_num == 0:
            num_neg = num_files
        if dir_num == 1:
            num_pos = num_files

        for filename in filenames:
            if (
                file_num_dir / num_neg
            ) < train_split:  # Ensure splits are the same, use num_neg
                tempdir = tempdir_train

                labels_train.append(
                    [str(file_num).zfill(4), dir_num]
                )  # NEGATIVES MUST COME FIRST FOR THIS TO WORK
                # Add labels for augmented images

                if dir_num == 1 and num_pos < num_neg:  # augment as needed
                    copy_rename_augment(
                        dir,
                        tempdir,
                        filename,
                        file_num,
                        augmentation_functions,
                    )
                    file_num += 1
                    file_num_dir += 1
                    labels_train.append([str(file_num).zfill(4), dir_num])
                    num_pos += 1

            elif (file_num_dir / num_neg) < test_split:
                tempdir = tempdir_test
                labels_test.append(
                    [str(file_num).zfill(4), dir_num]
                )  # NEGATIVES MUST COME FIRST FOR THIS TO WORK
            else:
                tempdir = tempdir_val
                labels_val.append([str(file_num).zfill(4), dir_num])
            copy_rename(
                dir,
                tempdir,
                filename,
                file_num,
            )
            file_num += 1
            file_num_dir += 1

    # Shuffle and write label data
    label_list = [
        (labels_train, tempdir_train, imgdir_train),
        (labels_test, tempdir_test, imgdir_test),
        (labels_val, tempdir_val, imgdir_val),
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
