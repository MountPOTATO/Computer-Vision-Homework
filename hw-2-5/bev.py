# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : bev.py
 # #@Author     : mount_potato
 # @Date        : 2022/5/18 17:21
"""

import cv2
import numpy as np

img=cv2.imread("img/data/1.pic.jpg")


h,w=img.shape[0],img.shape[1]


M=np.load("instrinsic/camera_matrix.npy")
dist=np.load("instrinsic/distortion_coefficient.npy")





newM,roi=cv2.getOptimalNewCameraMatrix(M,dist,(w,h),1,(w,h))

dst=cv2.undistort(img,M,dist,None,newM)


x,y,w,h=roi

dst=dst[y:y+h,x:x+w]
cv2.imwrite("img/output1.jpg",dst)

mapx,mapy=cv2.initUndistortRectifyMap(M,dist,None,newM,(w,h),5)
dst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imwrite("img/output2.jpg",dst)




