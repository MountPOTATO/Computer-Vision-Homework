# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : utils.py
 # #@Author     : mount_potato
 # @Date        : 2022/3/15 10:09 下午
"""
import numpy as np
import os
import uuid
from PIL import Image
import sys

# Tool box

def picture_to_array(pic, scale=1, gray=False):
    """
    generate the numpy array of an input picture with scale factor
    :param pic: the file path of the input picture
    :param scale: scale value, above 0
    :param gray: whether the picture is 3-channel or gray
    :return: the np.array of the scaled picture
    """
    img = Image.open(pic)

    # scale value check
    if scale <= 0:
        print("Error, scale must be non-negative.")
        sys.exit(-1)

    width = int(img.size[0] * scale)
    height = int(img.size[1] * scale)

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


def pic_rgb_to_grey(pic):
    """
    turning a 3-channel Image to gray
    :param pic: Image or np.array
    :return: gray array
    """
    r,g,b=pic[:,:,0],pic[:,:,1],pic[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b
