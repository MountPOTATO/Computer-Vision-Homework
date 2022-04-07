# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : sift.py
 # #@Author     : mount_potato
 # @Date        : 2022/4/5 15:08
"""


import cv2
import numpy as np


def get_keypoint_descriptor(gray_pic):
    sift=cv2.SIFT_create()
    key_points=sift.detect(gray_pic,None)
    #descriptor: num(kp)*128
    key_points,descriptor=sift.compute(gray_pic,key_points)
    return key_points, descriptor

def get_match_from_descriptor(descriptor1,descriptor2):
    # create brute-force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1,descriptor2,k=2)
    matches = sorted(matches,key=lambda x:x[0].distance/x[1].distance)
    # ratio test
    good = []

    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good





