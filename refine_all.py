#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
import cv2 as cv
import cv2 as cv
import numpy as np
import math
from scipy import optimize as opt


# 微调所有参数
def refinall_all_param(A, k, W, real_coor, pic_coor):
    # 整合参数
    P_init = compose_paramter_vector(A, k, W)

    M = len(real_coor)#M为图像数量
    N = len(real_coor[0])#N为每幅图像点数

    # 微调所有参数
    P = opt.leastsq(value,
                    P_init,
                    args=(W, real_coor, pic_coor),
                    Dfun=jacobian)[0]

    # raial_error表示利用标定后的参数计算得到的图像坐标与真实图像坐标点的平均像素距离
    error = value(P, W, real_coor, pic_coor)
    raial_error = [np.sqrt(error[2 * i] ** 2 + error[2 * i + 1] ** 2) for i in range(len(error) // 2)]


    #print("total max error:\t", np.max(raial_error))

    # 返回拆解后参数，分别为内参矩阵，畸变矫正系数，每幅图对应外参矩阵
    return np.max(raial_error), decompose_paramter_vector(P)


# 把所有参数整合到一个数组内
def compose_paramter_vector(A, k, W):#A为内参矩阵，k为畸变矩阵，W为多幅图像的外参矩阵列表
    alpha = np.array([A[0, 0], A[1, 1], A[0, 2], A[1, 2], k[0], k[1], k[2], k[3], k[4]])
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:, :3], (W[i])[:, 3]#外参系数包括旋转向量和平移向量

        # 旋转矩阵转换为一维向量形式
        zrou = to_rodrigues_vector(R)

        w = np.append(zrou, t)
        P = np.append(P, w)
    return P


# 分解参数集合，得到对应的内参，外参，畸变矫正系数
def decompose_paramter_vector(P):
    [alpha, beta, uc, vc, k0, k1, k2, k3, k4] = P[0:9]
    A = np.array([[alpha, 0, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    k = np.array([k0, k1, k2, k3, k4])
    W = []
    M = (len(P) - 9) // 6

    for i in range(M):
        m = 9 + 6 * i
        zrou = P[m:m + 3]
        t = (P[m + 3:m + 6]).reshape(3, -1)

        # 将旋转矩阵一维向量形式还原为矩阵形式

        R = (cv.Rodrigues(zrou))[0]

        #R = to_rotation_matrix(zrou)

        # 依次拼接每幅图的外参
        w = np.concatenate((R, t), axis=1)
        W.append(w)

    W = np.array(W)
    return A, k, W


# 返回从真实世界坐标映射的图像坐标
#函数输入为A为相机内参，W为相机外参，k为畸变向量，coor为实际物点坐标
#函数输出为，在给定内参、外参、畸变参数情况下，实际物点对应的像点的像素坐标
def get_single_project_coor(A, W, k, coor):#A为相机内参，W为相机外参，k为畸变向量，coor为实际物点坐标
    single_coor = np.array([coor[0], coor[1], coor[2], 1])#扩展三维坐标为其次坐标


    coor_norm = np.dot(W, single_coor)
    coor_norm /= coor_norm[-1]#计算得到在相机坐标系下，畸变发生前，相机z轴坐标归一后，的物点的坐标


    r = np.linalg.norm(coor_norm)#计算物点坐标的模长

    x_d = coor_norm[0] * (1 + k[0] * np.square(r) + k[1] * np.power(r, 4) + k[4] * np.power(4 ,6)) \
          + 2 * k[2] * coor_norm[0] * coor_norm[1] \
          + k[3] * (np.square(r) + 2.0 * np.square(coor_norm[0]))
    y_d = coor_norm[1] * (1 + k[0] * np.square(r) + k[1] * np.power(r, 4) + k[4] * np.power(4 ,6)) \
          + k[2] * (np.square(r) + 2.0 * np.square(coor_norm[1])) \
          + 2 * k[3] * coor_norm[0] * coor_norm[1]
    u = A[0, 0] * x_d + A[0, 2]
    v = A[1, 1] * y_d + A[1, 2]


    return np.array([u, v])


# 返回所有点的真实世界坐标映射到的图像坐标与真实图像坐标的残差
'''
此函数为差距计算函数，
输入为P，维度为9+图像数*6
输出为差距error_Y，维度为
'''
def value(P, org_W, X, Y_real):
    M = (len(P) - 9) // 6#M为图像数
    N = len(X[0])#N为每幅图的像点数

    A = np.array([#A为内参矩阵
        [P[0], 0, P[2]],
        [0, P[1], P[3]],
        [0, 0, 1]
    ])
    Y = np.array([])#用于存储理论图像像素点坐标

    for i in range(M):#计算M幅图
        m = 9 + 6 * i

        # 取出当前图像对应的外参
        w = P[m:m + 6]


        # 不用旋转矩阵的变换是因为会有精度损失
        '''
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)
        '''
        W = org_W[i]
        # 计算每幅图的坐标残差
        for j in range(N):
            Y = np.append(Y, get_single_project_coor(A, W, np.array([P[4], P[5], P[6], P[7], P[8]]), (X[i])[j]))#计算理论图像坐标

    error_Y = np.array(Y_real).reshape(-1) - Y#计算实际像素点坐标与理论计算像素点坐标的差

    return error_Y


# 计算对应jacobian矩阵
#函数输入为P计算函数输入项，X为图像数*图像点左边构成的矩阵
#函数输出为value输出向量的每个分量对于P的每个分量的求偏导数构成的jacobian矩阵
def jacobian(P, WW, X, Y_real):

    M = (len(P) - 9) // 6#图像数量

    N = len(X[0])#每幅图像点数

    K = len(P)#P为待优化向量，K为P的维度

    A = np.array([#A为内参矩阵
        [P[0], 0, P[2]],
        [0, P[1], P[3]],
        [0, 0, 1]
    ])

    res = np.array([])

    for i in range(M):#不同图像的循环
        m = 9 + 6 * i#定位外参信息

        w = P[m:m + 6]#w为外参信息，包括旋转和平移
        R = to_rotation_matrix(w[:3])#旋转矩阵
        t = w[3:].reshape(3, 1)#平移矩阵
        W = np.concatenate((R, t), axis=1)#平移旋转矩阵

        for j in range(N):#一幅图像中不同点的循环
            res = np.append(res, get_single_project_coor(A, W, np.array([P[4], P[5], P[6], P[7], P[8]]), (X[i])[j]))

    # 求得x, y方向对P[k]的偏导
    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, P[k])

    return J.T


# 将旋转矩阵分解为一个向量并返回，Rodrigues旋转向量与矩阵的变换,最后计算坐标时并未用到，因为会有精度损失
def to_rodrigues_vector(R):
    p = 0.5 * np.array([[R[2, 1] - R[1, 2]],
                        [R[0, 2] - R[2, 0]],
                        [R[1, 0] - R[0, 1]]])
    c = 0.5 * (np.trace(R) - 1)

    if np.linalg.norm(p) == 0:
        if c == 1:
            zrou = np.array([0, 0, 0])
        elif c == -1:
            R_plus = R + np.eye(3, dtype='float')

            norm_array = np.array([np.linalg.norm(R_plus[:, 0]),
                                   np.linalg.norm(R_plus[:, 1]),
                                   np.linalg.norm(R_plus[:, 2])])
            v = R_plus[:, np.where(norm_array == max(norm_array))]
            u = v / np.linalg.norm(v)
            if u[0] < 0 or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] and u[0] == 0 and u[2] < 0):
                u = -u
            zrou = math.pi * u
        else:
            zrou = []
    else:
        u = p / np.linalg.norm(p)
        theata = math.atan2(np.linalg.norm(p), c)
        zrou = theata * u

    return zrou


# 把旋转矩阵的一维向量形式还原为旋转矩阵并返回
def to_rotation_matrix(zrou):

    theta = np.linalg.norm(zrou)
    zrou_prime = zrou / theta

    W = np.array([[0, -zrou_prime[2], zrou_prime[1]],
                  [zrou_prime[2], 0, -zrou_prime[0]],
                  [-zrou_prime[1], zrou_prime[0], 0]])
    R = np.eye(3, dtype='float') + W * math.sin(theta) + np.dot(W, W) * (1 - math.cos(theta))


    return R