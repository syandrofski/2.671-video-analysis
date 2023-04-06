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

def angles_to_hor(adv_struct, centers):
    frames, headers, points = adv_struct.shape
    xs = adv_struct[:, 0, :]
    ys = adv_struct[:, 1, :]
    for center in centers:
        relatives_x = np.zeros((xs.shape[0], 1, xs.shape[1]))
        relatives_y = np.zeros((xs.shape[0], 1, xs.shape[1]))
        relatives_x[:, 0, :] = np.tile(xs[:, center:center+1], (1, points)) - xs
        relatives_y[:, 0, :] = np.tile(ys[:, center:center+1], (1, points)) - ys

        angs = np.zeros_like(relatives_x)
        for i in range(xs.shape[1]):
            angs[:, 0, i] = angle(relatives_x[:, 0, i], relatives_y[:, 0, i])
        adv_struct = np.hstack((adv_struct, np.reshape(angs, (xs.shape[0], 1, xs.shape[1]))))
    return adv_struct
def main():
    pt = 3
    pts_x, pts_y = vp.track('test.mp4', 'y', pt)
    stats = pd.DataFrame(np.empty((pts_x.shape[0], pt)))
    for i in range(pt):
        if not i == pt-1:
            stats[i] = 0

if __name__ == '__main__':
    main()