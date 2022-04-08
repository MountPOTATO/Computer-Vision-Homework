# -*- coding: UTF-8 -*-
"""
 # @Project     : Computer-Vision-Homework
 # @File        : main.py.py
 # #@Author     : mount_potato
 # @Date        : 2022/3/24 10:44 上午
"""

import getopt
from utils import *
from sift import *


def main(argv):
    first_pic = ""
    second_pic = ""

    try:
        # get arguments from python command
        opts, args = getopt.getopt(argv, "hf:s:", ["help", "first=", "second="])
    except getopt.GetoptError:
        print("#! Error command form.")
        sys.exit(2)
    for key, value in opts:
        if key in ("-h", "--help"):
            # command help guideline
            print("TODO:Help usage")
            sys.exit(0)
        elif key in ("-f", "--first"):
            first_pic = value
        elif key in ("-s", "--second"):
            second_pic = value
        else:
            print("#! Error command form.")
            sys.exit(2)

    # main process

    pic1 = picture_to_array(first_pic, gray=False)
    pic2 = picture_to_array(second_pic, gray=False)

    y_size = pic1.shape[0] // 2
    x_size = pic1.shape[1] // 2

    #resize full-color pic
    pic1 = cv2.resize(pic1, (x_size, y_size))
    pic2 = cv2.resize(pic2, (x_size, y_size))

    gray_pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

    # get keypoints and descriptor vector
    keypoints1, descriptor1 = get_keypoint_descriptor(gray_pic1)
    keypoints2, descriptor2 = get_keypoint_descriptor(gray_pic2)

    # display keypoint result
    pic1_with_key_points = cv2.drawKeypoints(pic1, keypoints1, None)
    pic2_with_key_points = cv2.drawKeypoints(pic2, keypoints2, None)
    show_pic_from_array(pic1_with_key_points, save=False)
    show_pic_from_array(pic2_with_key_points, save=False)

    # get matches from the descriptor value
    match = get_match_from_descriptor(descriptor1, descriptor2)

    match_picture = cv2.drawMatches(pic1_with_key_points,
                                    keypoints1,
                                    pic2_with_key_points,
                                    keypoints2,
                                    match,
                                    None,
                                    flags=2)

    # need at least 4 matches
    if len(match_picture) > 4:
        points1 = np.float32([keypoints1[i.queryIdx].pt for i in match]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[i.trainIdx].pt for i in match]).reshape(-1, 1, 2)

        # get homography
        H, status = cv2.findHomography(points1,
                                       points2,
                                       cv2.RANSAC,
                                       ransacReprojThreshold=4)

        panoramic = cv2.warpPerspective(pic1,
                                        H,
                                        (pic1.shape[1] + pic2.shape[1], pic1.shape[0]))
        panoramic[0:pic2.shape[0], 0:pic2.shape[1]] = pic2

        show_pic_from_array(match_picture)

        show_pic_from_array(panoramic, save=True)


if __name__ == "__main__":
    main(sys.argv[1:])
