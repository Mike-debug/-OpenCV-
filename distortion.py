#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import numpy as np


# 返回畸变矫正系数k0,k1,p0,p1
def get_distortion(intrinsic_param, extrinsic_param, pic_coor, real_coor):
    D = []
    d = []
    for i in range(len(pic_coor)):
        for j in range(len(pic_coor[i])):
            # 转换为齐次坐标
            single_coor = np.array([(real_coor[i])[j, 0], (real_coor[i])[j, 1], 0, 1])

            # 利用现有内参及外参求出估计图像坐标
            u = np.dot(np.dot(intrinsic_param, extrinsic_param[i]), single_coor)
            [u_estim, v_estim] = [u[0] / u[2], u[1] / u[2]]

            #最终计算得到的coor_norm是在相机坐标下，无畸变前，相机坐标z轴归一后，的物点的坐标
            coor_norm = np.dot(extrinsic_param[i], single_coor)
            coor_norm /= coor_norm[-1]

            # r = np.linalg.norm((real_coor[i])[j])
            r = np.linalg.norm(coor_norm)

            D.append(np.array([(u_estim - intrinsic_param[0, 2]) * r ** 2,
                               (u_estim - intrinsic_param[0, 2]) * r ** 4,
                               2.0*intrinsic_param[0,0]*coor_norm[0]*coor_norm[1]+intrinsic_param[0,1]*(r**2+2.0*np.square(coor_norm[1])),
                               intrinsic_param[0,0]*(np.square(r)+2.0*np.square(coor_norm[0]))+2.0*intrinsic_param[0,1]*coor_norm[0]*coor_norm[1],
                               (u_estim - intrinsic_param[0, 2]) * r ** 6
                               ]))
            D.append(np.array([(v_estim - intrinsic_param[1, 2]) * r ** 2,
                               (v_estim - intrinsic_param[1, 2]) * r ** 4,
                               intrinsic_param[1,1]*(np.square(r)+2.0*np.square(coor_norm[1])),
                               2.0*intrinsic_param[1,1]*coor_norm[0]*coor_norm[1],
                               (u_estim - intrinsic_param[1, 2]) * r ** 6
                               ]))

            # 求出估计坐标与真实坐标的残差
            d.append(pic_coor[i][j, 0] - u_estim)
            d.append(pic_coor[i][j, 1] - v_estim)

    #利用SVD求解D * k = d中的k
    U, S, Vh=np.linalg.svd(D, full_matrices=False)

    temp_S = np.identity(len(S))
    for i in range(len(S)):
        temp_S[i, i] = S[i]

    temp_res = np.dot(Vh.transpose(), np.linalg.inv(temp_S))
    temp_res_res = np.dot(temp_res, U.transpose())

    k = np.dot(temp_res_res, d)

    return k