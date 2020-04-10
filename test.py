#本模块为过程中测试使用
import math
import cv2 as cv
import numpy as np
import cython
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from timeit import timeit as timeit
from scipy.optimize import basinhopping as bh
s = [1,2, 3,4]
di = np.identity(len(s))
for i in range(len(s)):
    di[i,i]=s[i]
print(di)
