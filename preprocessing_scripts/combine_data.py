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


def copy_rename(old_dir: str, new_dir: str, filename: str, file_num: int):
    """Copy from one path to another and rename according to file_num"""
    old_path = join(old_dir, filename)
    new_path = join(new_dir, str(file_num).zfill(4) + filename[-4:])
    copy(old_path, new_path)


def combine(dirs: list[str]):
    """Combine positive and negative image dirs into one, shuffle, and generate label csv"""

    # Initialize file counter and label list
    file_num = 1
    labels = []
    tempdir = "data/images_temp"
    imgdir = "data/images"

    # Clear image folders if exists
    rmtree("data", ignore_errors=True)
    mkdir("data")
    mkdir(imgdir)
    mkdir(tempdir)  # for copying data

    # Iterate over each dir and copy
    for dir_num, dir in enumerate(dirs):
        filenames = listdir(dir)
        for filename in filenames:
            copy_rename(dir, tempdir, filename, file_num)
            labels.append(
                [str(file_num).zfill(4), dir_num, join(dir, filename)]
            )  # NEGATIVES MUST COME FIRST FOR THIS TO WORK
            file_num += 1

    # Shuffle
    shuffle(labels)
    for ind, label in enumerate(labels):
        try:
            copy_rename(
                tempdir, imgdir, label[0] + ".jpg", ind
            )  # copy files in shuffled order
        except FileNotFoundError:
            copy_rename(
                tempdir, imgdir, label[0] + ".png", ind
            )  # just some sketchy error handling to accound for multiple data types

    labels = [
        [str(ind).zfill(4), label[1], label[2]] for ind, label in enumerate(labels)
    ]  # rename labels based on their new index

    # Write label data
    with open("data/image_labels.csv", "w") as label_file:
        filewriter = csv.writer(label_file)
        filewriter.writerows(labels)

    # delete tempdir
    rmtree(tempdir, ignore_errors=True)
