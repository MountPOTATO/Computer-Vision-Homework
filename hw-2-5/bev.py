# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : bev.py
 # #@Author     : mount_potato
 # @Date        : 2022/5/18 17:21
"""

import cv2
import numpy as np

img=cv2.imread("img/input.jpg")

h,w=img.shape[0],img.shape[1]



# load intrinsic
K=np.load("intrinsic/camera_matrix.npy")
distortion=np.load("intrinsic/distortion_coefficient.npy")

# undistort,get
undst=cv2.undistort(img,K,distortion,None)

# cv2.imwrite("img/undistort.jpg",undst)


# wcs
w_point = np.zeros((9 * 6, 3), np.float32)
w_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# find chessboard 4 corners on undistorted image
gray = cv2.cvtColor(undst, cv2.COLOR_BGR2GRAY)
_, corners = cv2.findChessboardCorners(gray, (9, 6), None)

corner_point=np.array([corners[0,:],corners[8,:],corners[-6,:],corners[-1,:]])

# corner points on wcs as pc
pw_list = np.array([w_point[0, :][:-1], w_point[8, :][:-1], w_point[-6, :][:-1], w_point[-1, :][:-1]])


# bird eye view coord pb
# assuming M/H=70, N/2=200, M/2=600
pb_list = np.array([w_point[0,:][:-1],w_point[8,:][:-1],w_point[-6,:][:-1],w_point[-1,:][:-1]])
for index,pb in enumerate(pb_list):
    pb[0]= pw_list[index][0] * 70 + 200
    pb[1]= pw_list[index][1] * 70 + 600


# generate transformation matrix:
trans=cv2.getPerspectiveTransform(corner_point,pb_list)


out_img = cv2.warpPerspective(undst, trans, (undst.shape[0],undst.shape[1]))


cv2.imwrite("img/output_bev.jpg",out_img)










