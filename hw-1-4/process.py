# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : process.py
 # #@Author     : mount_potato
 # @Date        : 2022/3/16 8:19 下午
"""

from math import sqrt, hypot
from numpy import arccos
import numpy as np
from itertools import combinations
from tqdm import tqdm
import cv2



def pic_LoG_filter(sigma):
    #according to 3-sigma principle,sigma_times should be 3*2=6
    sigma_times=6
    kernel_size=np.ceil(sigma*sigma_times)
    y,x=np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]


    # applying LoG
    x_pow2=x*x
    y_pow2=y*y
    sigma_pow2=sigma**2
    exp_x2 = np.exp(-(x_pow2 / (2. * sigma_pow2)))
    exp_y2 = np.exp(-(y_pow2 / (2. * sigma_pow2)))

    #broadcasting
    return (1/(2*np.pi*sigma**4))*(-(2*sigma_pow2)+(x_pow2+y_pow2))*(exp_x2*exp_y2)



def convolution(kernel, origin_img):
    """
    hand-write convolution
    :param kernel: convolution kernel
    :param origin_img: origin input img
    :return: the convolution result
    """

    y_k,x_k=kernel.shape
    y,x=origin_img.shape
    origin_img=np.pad(origin_img,(0,y_k-1))
    new_img=np.zeros((y,x),dtype=np.float)
    for i in range(y):
        for j in range(x):
            img_slice=origin_img[i:i+y_k,j:j+x_k]
            new_img[i][j]=np.sum(np.multiply(kernel,img_slice))
    return new_img

#
# utils.show_pic_from_array(pic_LoG_filter(5))


def pic_convolution(origin_img):
    """
    the LoG convolution process
    :param origin_img: origin input img
    :return: an np.array of the convolution result
    """
    conv_img_list=[]
    conv_img_num=10
    sigma_ratio=1.41
    sigma=1
    print("start convolution...")
    for i in tqdm(range(conv_img_num)):
        sigma*=sigma_ratio
        kernel_LoG=pic_LoG_filter(sigma)

        # using cv2
        conv_result=cv2.filter2D(origin_img, -1, kernel_LoG)

        # hand-write convolution function(slow)
        # conv_result = convolution(kernel_LoG,origin_img)

        # utils.show_pic_from_array(conv_result)

        # use np.pad to fill the edge of the image
        image=np.pad(conv_result,((1,1),(1,1)),'constant')

        conv_img_list.append(image)
    return np.array(conv_img_list)




def blob_maximum_extract(conv_img_list):
    """
    given a convolution result list , apply LoG to detect maximum value
    :param conv_img_list: the output of pic_convolution
    :return: available points with sigma
    """
    z_set=set()
    blobs=[]
    if len(conv_img_list)==0:
        print("Error: blob_maximum_extract should be applied after convolution")
        exit(-1)
    (h,w)=conv_img_list[0].shape
    for i in range(1,h-5):
        for j in range(1,w-5):
            # for k in range(0,l-3):
                kernel_img=conv_img_list[:,i-1:i+2,j-1:j+2]
                threshold=np.max(kernel_img)
                if threshold>=50:
                    max_index=kernel_img.argmax()
                    z,y,x=np.unravel_index(max_index,kernel_img.shape)
                    blobs.append((i+y-1,j+x-1,1.41**z))
                    z_set.add(1.41**z)

    print("found {} blobs".format(len(blobs)))

    # print("sigma value list: ",z_list)

    return np.array(list(set(blobs)))




# from scipy
def blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area.
    Parameters
    ----------
    blob1 : sequence
        A sequence of ``(y,x,sigma)``, where ``x,y`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.
    blob2 : sequence
        A sequence of ``(y,x,sigma)``, where ``x,y`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.
    Returns
    -------
    f : float
        Fraction of overlapped area.
    """
    root2 = sqrt(2)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[2] * root2
    r2 = blob2[2] * root2

    d = hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])

    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = arccos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = arccos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d))

    return area / (np.pi * (min(r1, r2) ** 2))


def remove_blobs(blobs, threshold):
    """
    removing redundant blobs
    :param blobs: the output of blob_maximum_extract
    :param threshold: the area's maximum overlap rate
    :return: the final blobs
    """
    print("removing unnecessary blobs,this may take a while...")

    blob_pairs=[(blob1,blob2) for blob1,blob2 in combinations(blobs, 2) ]

    pairs_len=len(blob_pairs)
    for i in tqdm(range(pairs_len)):
        (blob1,blob2)=blob_pairs[i]
        overlap_sum=blob_overlap(blob1,blob2)
        if overlap_sum>threshold:
            if blob1[2]>blob2[2]:
                blob2[2]=0
            else:
                blob1[2]=0

    final_blob=[blob for blob in blobs if blob[2]>0]
    print("remain blobs: ",len(final_blob))
    return np.array(final_blob)

