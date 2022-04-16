"""
This script is intended to copy image data and 
image annotations from the JHU Crowd++ dataset and 
modify the image annotations for the purpose of 
our own investigation.

Assumptions:
 - The original data will be stored in a parent folder 
    `jhu_crowd_v2.0` containing `test`, `train`, and `val` subfolders,
    each containig subfolders `gt` and `images` as well as an `image_labels.txt` in csv format
 - The copied `data` folder will include corresponding copied `images` 
    and `image_labels.txt`.
 - All images are jpg
 - All JHU Crowd++ images are negative cases (do not include social distancing)
 - Images that are not well suited for our model due to the following criteria will be excluded manually
    - No or few people in image
    - Socially distanced crowd
    - Image taken above crowd rather than horizontal to crowd
    - Severe blurring
    - Severe weather conditions
    - Difficulty discerning people in photo
 - JHU Crowd++ has 1600 test images, 2272 train images, 501 validation images
 - We use the first 100 test images, 1000 train images, 100 validation images (excluding any manual removals)
"""

import csv
from shutil import copy, rmtree
from os.path import join
from os import mkdir
from tensorflow.keras.preprocessing.image import load_img, save_img


def img_rename(file_num: int):
    return str(file_num).zfill(4)


def get_negative_data(old_dir: str, new_dir: str):
    jhu_parent_dir = old_dir
    new_parent_dir = new_dir
    # sub_dirs = [["test", 100], ["train", 1000], ["val", 100]]
    sub_dirs = [["train", 800], ["test", 200], ["val", 200]]
    label_filename = "image_labels.txt"

    # manual removals
    remove = {
        "test": [
            5,
            41,
            80,
            87,
            98,
            104,
            123,
            129,
            132,
            140,
            157,
            223,
            226,
            236,
            240,
            244,
            256,
            262,
            299,
            303,
            332,
            392,
            396,
            402,
            409,
            410,
            421,
            433,
            447,
            471,
            473,
            479,
            515,
            527,
            535,
            538,
            571,
            576,
            577,
        ],
        "train": [
            37,
            39,
            44,
            55,
            57,
            77,
            82,
            99,
            100,
            109,
            135,
            137,
            167,
            187,
            189,
            194,
            217,
            233,
            247,
            255,
            270,
            271,
            290,
            302,
            313,
            320,
            352,
            365,
            368,
            370,
            373,
            380,
            387,
            404,
            405,
            432,
            436,
            437,
            444,
            448,
            450,
            452,
            461,
            475,
            481,
            539,
            560,
            655,
            659,
            661,
            662,
            668,
            702,
            708,
            710,
            729,
            738,
            746,
            751,
            757,
            769,
            816,
            848,
            863,
            893,
            913,
            918,
            921,
            932,
            962,
            967,
            1002,
            1004,
            1020,
            1023,
            1024,
            1060,
            1085,
            1105,
            1126,
            1176,
            1177,
            1180,
            1183,
            1184,
            1186,
            1206,
            1213,
            1214,
            1247,
            1260,
            1268,
            1270,
            1287,
            1289,
            1307,
            1312,
            1324,
            1348,
            1353,
            1356,
            1378,
            1405,
            1427,
            1443,
            1468,
            1469,
            1472,
            1495,
            1504,
            1505,
            1509,
            1543,
            1564,
            1576,
            1611,
            1626,
            1639,
            1640,
            1643,
            1646,
            1649,
            1654,
            1694,
            1698,
            1700,
            1707,
            1709,
            1735,
            1743,
            1786,
            1791,
            1792,
            1804,
            1805,
            1806,
            1821,
            1844,
            1850,
            1855,
            1879,
            1887,
            1888,
            1910,
            1918,
            1928,
            1929,
            1948,
            1954,
            2020,
            2046,
            2067,
            2073,
            2075,
            2078,
            2085,
            2095,
            2126,
            2129,
            2139,
            2142,
            2190,
            2202,
            2234,
            2251,
            2259,
            2276,
            2281,
            2305,
            2308,
            2328,
            2346,
            2349,
            2357,
            2362,
            2364,
            2369,
            2376,
            2381,
            2414,
            2425,
            2457,
            2465,
            2468,
            2474,
            2480,
            2505,
            2512,
            2535,
            2539,
            2545,
            2559,
            2569,
            2595,
            2610,
            2620,
            2625,
            2645,
            2651,
            2657,
            2663,
            2670,
            2682,
            2716,
            2730,
            2741,
            2745,
            2762,
            2775,
            2779,
            2813,
            2841,
            2851,
            2868,
        ],
        "val": [
            11,
            92,
            105,
            134,
            213,
            348,
            349,
            508,
            512,
            530,
            546,
            612,
            634,
            653,
            745,
            748,
            761,
            924,
            951,
            986,
            1107,
            1164,
            1166,
            1209,
            1257,
            1288,
            1387,
            1404,
            1511,
            1516,
            1566,
        ],
    }

    # Clear image folder if it exists
    new_path = f"./{new_parent_dir}"
    rmtree(join(new_path), ignore_errors=True)
    mkdir(join(new_path))

    total_img_copy = 0
    for sub_dir_data in sub_dirs:
        # Define paths
        sub_dir = sub_dir_data[0]
        num_images = sub_dir_data[1]
        remove_list = remove.get(sub_dir)
        remove_list = [str(val).zfill(4) for val in remove_list]  # format remove list
        jhu_path = f"./{jhu_parent_dir}/{sub_dir}"

        # Copy label data (except for manually removed entries)
        with open(join(jhu_path, label_filename), "r") as label_file:

            jhu_labels = csv.reader(label_file)

            num_copy = 0

            for label in jhu_labels:

                if label[0] not in remove_list:  # check that image isn't removed
                    total_img_copy += 1
                    fn = label[0]
                    fn = img_rename(
                        total_img_copy
                    )  # OPTIONAL - Rename the copied images sequentially for our own purposes
                    save_img(
                        join(new_path, f"{fn}.jpg"),
                        load_img(
                            join(jhu_path, f"images/{label[0]}.jpg"),
                            target_size=(224, 224),
                        ),
                    )
                    num_copy += 1

                if num_copy > num_images - 1:  # check if we have enough images
                    break
