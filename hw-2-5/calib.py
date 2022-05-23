# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : calib.py
 # #@Author     : mount_potato
 # @Date        : 2022/5/18 15:12
"""
import glob

import numpy as np
import cv2

images=glob.glob("img/data/*.jpg")





board_size=(9,6)
img_size=(640,480)

#wcs
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

obj_points=[]  #wcs的3D世界中的实际点坐标
img_points=[]  #图像平面坐标，由obj_points通过矩阵变换得到

for path in images:
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size=gray.shape[::-1]

    # 角点检测
    ret, corners=cv2.findChessboardCorners(gray,board_size,None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)


# 相机标定
ret,K,dist,rvecs,tvecs=cv2.calibrateCamera(obj_points,
                                           img_points,
                                           img_size,
                                           None,None)



print("K:",K)
print("Distortion:",dist)

np.save("intrinsic/camera_matrix", K)
np.save("intrinsic/distortion_coefficient", dist)



