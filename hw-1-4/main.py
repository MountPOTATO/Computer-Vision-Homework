# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : main.py.py
 # #@Author     : mount_potato
 # @Date        : 2022/3/15 9:23 下午
"""


import getopt
from utils import *
from process import *
import matplotlib.pyplot as plt
import time


def main(argv):
    input_pic = ""
    output_pic = ""
    scale = 0
    try:
        # get arguments from python command
        opts, args = getopt.getopt(argv, "hi:s:", ["help", "input=", "scale="])
    except getopt.GetoptError:
        print("#! Error command form.")
        sys.exit(2)
    for key, value in opts:
        if key in ("-h", "--help"):
            # command help guideline
            print("TODO:Help usage")
            sys.exit(0)
        elif key in ("-i", "--input"):
            input_pic = value
        elif key in ("-s", "--scale"):
            try:
                scale = float(value)
                if scale <= 0:
                    raise ValueError
                else:
                    break
            except ValueError:
                print("Error: scale should be a positive float-type number")

    # 入口图像处理

    # show_pic_from_array(picture_to_array(input_pic))
    pic=picture_to_array(input_pic,scale=scale,gray=True)
    color_pic = picture_to_array(input_pic, scale=scale, gray=False)
    # show_pic_from_array(convolution(pic_LoG_filter(4),a),save=False)
    # conv_array=convolution(pic_LoG_filter(4), a)
    coord=blob_maximum_extract(pic_convolution(pic))
    coord=remove_blobs(coord,0.7)

    fig, ax = plt.subplots()

    (origin_y,origin_x)=pic.shape
    ax.imshow(color_pic, interpolation='nearest')
    for blob in coord:
        y, x, r = blob
        if y<origin_y and x<origin_x:
            c = plt.Circle((x, y), r * 1.41, color='red', linewidth=1.0, fill=False)
            ax.add_patch(c)
    ax.plot()

    now=time.localtime()
    newt=time.strftime("%Y-%M-%d-%H_%M_%S",now)

    plt.savefig("result/"+newt+"scale-"+str(scale)+".png")
    # draw_pic_from_array(picture_to_array(input_pic,scale=scale),save=True)


if __name__ == "__main__":
    main(sys.argv[1:])
