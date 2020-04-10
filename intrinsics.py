#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import numpy as np


# 返回ij位置对应的v向量
def create_v(p, q, H):
    H = H.reshape(3, 3)
    return np.array([
        H[0, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])



# 返回相机内参矩阵A
def get_intrinsics_param(H):
    # 构建V矩阵
    V = np.array([])



    for i in range(len(H)):
        V = np.append(V, np.array([create_v(0, 1, H[i]), create_v(0, 0, H[i]) - create_v(1, 1, H[i])]))

    # 求解V*b = 0中的b
    U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 5)))
    # 最小的奇异值对应的奇异向量,S求出来按大小排列的，最后的最小
    b = VT[-1]

    vc = -(b[0] * b[3]) / (b[0] * b[1])
    lamda = b[4] - (np.square(b[2]) - vc * b[0] * b[3]) / b[0]
    alpha = np.sqrt(lamda / b[0])
    beta = np.sqrt(lamda / b[1])
    gamma = 0
    uc = -b[2] * np.square(alpha) / lamda

    intrinsic_mat = np.array([
        [alpha, gamma, uc],
        [0, beta, vc],
        [0, 0, 1]
    ])


    return intrinsic_mat