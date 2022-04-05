# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : utils.py.py
 # #@Author     : mount_potato
 # @Date        : 2022/4/5 15:00
"""

import numpy as np
import os
import uuid
from PIL import Image
import sys


def picture_to_array(pic, gray=False):
    """
    generate the numpy array of an input picture with scale factor
    :param pic: the file path of the input picture
    :param gray: whether the picture is 3-channel or gray
    :return: the np.array of the scaled picture
    """
    img = Image.open(pic)


    width = int(img.size[0])
    height = int(img.size[1])

    # avoid situations where the scale isn't huge enough
    if width <= 0 or height <= 0:
        print("Error: the scale is too small.")
        sys.exit(1)

    if gray:
        return np.array(img.resize((width, height)).convert('L'))
    else:
        return np.array(img.resize((width, height)))

def show_pic_from_array(pic_arr, save=False):
    """
    open picture with the np.array of an picture
    :param pic_arr: the np.array of an picture
    :param save: boolean. If True, save the picture to "./result"
    """
    img = Image.fromarray(pic_arr.astype('uint8'))
    img.show()

    if save:
        # generate random file name
        uuid_str = uuid.uuid4().hex
        # get current file directory
        cur_dir = os.path.split(os.path.realpath(__file__))[0]

        target_dir = cur_dir + "/result"

        # new result folder if don't have one
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        img.save(target_dir + "/" + uuid_str + ".jpg")


