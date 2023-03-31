import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt
from math import sin, cos, tan, atan, sqrt
import time
import copy
import video_parse as vp


def angle(xs, ys):
    return(np.arctan(ys/xs))


def main():
    pt = 3
    pts_x, pts_y = vp.track('test.mp4', 'y', pt)
    stats = pd.DataFrame(np.empty((pts_x.shape[0], pt)))
    for i in range(pt):
        if not i == pt-1:
            stats[i] = 0

if __name__ == '__main__':
    #main()
    #print(angle(np.array([5]), np.array([5])))
    test1 = np.array([[1, 2], [3, 4]])
    test2 = np.array([5, (6, 7)])
    print(test1.shape, test2.shape)
    print(np.vstack((test1, test2)))