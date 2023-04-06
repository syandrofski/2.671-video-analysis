import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt
from math import sin, cos, tan, atan, sqrt, pi
import time
import copy
import video_parse as vp
import window_wrapper as ww


def angle(xs, ys):
    return np.arctan2(ys, xs)*180/pi

def angles_to_hor(adv_struct):
    xs = adv_struct[:, 0, :]
    ys = adv_struct[:, 0, :]
    angs = np.zeros_like(xs)
    for i in range(xs.shape[1]):
        angs[:, i] = angle(xs[:, i], ys[:, i])
    angs = np.reshape(angs, (xs.shape[0], 1, xs.shape[1]))
    return np.hstack((adv_struct, angs))
def main():
    pt = 3
    pts_x, pts_y = vp.track('test.mp4', 'y', pt)
    stats = pd.DataFrame(np.empty((pts_x.shape[0], pt)))
    for i in range(pt):
        if not i == pt-1:
            stats[i] = 0

if __name__ == '__main__':
    main()