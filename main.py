#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import cv2 as cv
import numpy as np
import os
import glob
from homography import get_homography
from intrinsics import get_intrinsics_param
from extrinsics import get_extrinsics_param
from distortion import get_distortion
from refine_all import refinall_all_param


def calibrate(pic_points, real_points_x_y):

    # 求单应矩阵
    H = get_homography(pic_points, real_points_x_y)

    # 求内参
    intrinsics_param = get_intrinsics_param(H)

    # 求对应每幅图外参
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)

    # 畸变矫正
    k = get_distortion(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)

    # 微调所有参数
    max_raial_error, [new_intrinsics_param, new_k, new_extrinsics_param] = refinall_all_param(intrinsics_param, k, extrinsics_param, real_points, pic_points)
    #print("total max error:\t", max_raial_error)
    #print("intrinsics_parm:\t", new_intrinsics_param)
    #print("distortionk:\t", new_k)
    #print("extrinsics_parm:\t", new_extrinsics_param)
    MyRecord.write("total max error:\n")
    MyRecord.write(str(max_raial_error))
    MyRecord.write("\n")
    MyRecord.write("intrinsics_parm:\n")
    MyRecord.write(str(new_intrinsics_param))
    MyRecord.write("\n")
    MyRecord.write("distortionk:\n")
    MyRecord.write(str(new_k))
    MyRecord.write("\n")
    MyRecord.write("extrinsics_parm:\n")
    MyRecord.write(str(new_extrinsics_param))
    MyRecord.write("\n")

if __name__ == "__main__":
    # 定义记录输出数据的文件目录
    record_dir = r"record"
    MyRecord = open(os.path.join(record_dir, r"myrecord.txt"), "w")
    StandardRecord = open(os.path.join(record_dir, r"standardrecord.txt"), "w")
    file_dir = r'images'
    # 标定所用图像
    pic_name = os.listdir(file_dir)

    # 由于棋盘为二维平面，设定世界坐标系在棋盘上，一个单位代表一个棋盘宽度，产生世界坐标系三维坐标
    cross_corners = [9, 6]  # 棋盘方块交界点排列
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor[:, :2] = np.mgrid[0:9*30:30, 0:6*30:30].T.reshape(-1, 2)


    real_points = []
    real_points_x_y = []
    pic_points = []

    flag = True
    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)

        # 寻找到棋盘角点
        succ, pic_coor = cv.findChessboardCorners(pic_data, (cross_corners[0], cross_corners[1]), None)


        if succ:

            # 灰度化
            pic_data = cv.cvtColor(pic_data, cv.COLOR_RGB2GRAY)

            #获取图像大小
            if flag:
                size = pic_data.shape[::-1]
                flag = False

            # 亚像素精细化
            cv.cornerSubPix(
                pic_data,
                pic_coor,
                (11, 11),
                (-1, -1),
                (cv.TermCriteria_MAX_ITER + cv.TermCriteria_EPS, 30, 0.01)
            )


            # 添加每幅图的对应3D-2D坐标
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)

            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])

            pic_points1 = pic_points.copy()
            real_points_x_y1 = real_points_x_y.copy()

    #调用自己定义的标定函数
    calibrate(pic_points, real_points_x_y)
    MyRecord.close()

    #使用OpenCV库函数标定
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(real_points, pic_points, size, None, None)


    W = []#外参矩阵
    for i in range(len(tvecs)):
        Rotate_M = (cv.Rodrigues(rvecs[i]))[0]
        w = np.concatenate((Rotate_M, tvecs[i]), axis=1)
        w = np.array(w)
        W.append(w)

    StandardRecord.write("Reprojection error:\n")
    StandardRecord.write(str(ret))
    StandardRecord.write("\n")
    StandardRecord.write("intrinsics_parm:\n")
    StandardRecord.write(str(mtx))
    StandardRecord.write("\n")
    StandardRecord.write("distortionk:\n")
    StandardRecord.write(str(dist))
    StandardRecord.write("\n")
    StandardRecord.write("extrinsics_parm:\n")
    for i in range(len(W)):
        StandardRecord.write(str(W[i]))
        StandardRecord.write("\n\n")
