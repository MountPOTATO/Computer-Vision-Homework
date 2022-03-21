# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : process.py
 # #@Author     : mount_potato
 # @Date        : 2022/3/16 8:19 下午
"""


import numpy as np
import scipy
import matplotlib.image as mp
import cv2

import utils
def pic_LoG_filter(sigma):
    #according to 3-sigma principle,sigma_times should be 3*2=6
    sigma_times=6
    kernel_size=np.ceil(sigma*sigma_times)
    y,x=np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]

    # applying LoG
    x_pow2=x*x
    y_pow2=y*y
    sigma_pow2=sigma**2
    #broadcasting
    return -(1/(np.pi*sigma_pow2**2))*(1-(x_pow2+y_pow2)/(2*sigma_pow2))*np.exp(-(x_pow2+y_pow2)/(2*sigma_pow2))



def convolution(kernel, origin_img):
    """

    :param kernel:
    :param origin_img:
    :return:
    """
    y_k,x_k=kernel.shape
    y,x=origin_img.shape
    new_img=[]
    for i in range(y-y_k):
        line=[]
        for j in range(x-x_k):
            img_slice=origin_img[i:i+y_k,j:j+x_k]
            line.append(np.sum(np.multiply(kernel,img_slice)))
        new_img.append(line)
    return np.array(new_img)

#
# utils.show_pic_from_array(pic_LoG_filter(5))


def pic_convolution(origin_img):
    conv_img_list=[]
    conv_img_num=5
    sigma_ratio=1.2
    sigma=1
    for i in range(conv_img_num):
        sigma*=sigma_ratio
        kernel_LoG=pic_LoG_filter(sigma)
        # use np.pad to fill the edge of the image
        conv_result=cv2.filter2D(origin_img, -1, kernel_LoG)
        # conv_result = convolution(kernel_LoG,origin_img)
        # utils.show_pic_from_array(conv_result)
        # print(conv_result)
        image=np.pad(conv_result,((1,1),(1,1)),'constant')

        conv_img_list.append(np.array(image))
    return np.array(conv_img_list)

def blob_maximum_extract(conv_img_list):
    max_coord=[]
    if len(conv_img_list)==0:
        print("Error: blob_maximum_extract should be applied after convolution")
        exit(-1)
    (h,w)=conv_img_list[0].shape
    for i in range(1,h):
        for j in range(1,w):
            kernel_img=conv_img_list[:,i-1:i+2,j-1:j+2]
            threshold=np.max(kernel_img)
            if threshold>=6:
                max_index=np.array(kernel_img).argmax()
                z,x,y=np.unravel_index(max_index,kernel_img.shape)
                #DoG
                max_coord.append((i+x-1,j+y-1,2**z))
    print(len(max_coord))


    return np.array(list(set(max_coord)))










