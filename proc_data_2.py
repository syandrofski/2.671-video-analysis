import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from math import pi

def compute_angle(pts1, pts2, pts3):
    return pts1[:, 0] + pts2[:, 0] + pts3[:, 0]

def process_struct(adv_struct):
    frames, cats, points = adv_struct.shape
    angles = []
    for p in range(points-2):
        pt1 = adv_struct[:, 0:2, p]
        pt2 = adv_struct[:, 0:2, p+1]
        pt3 = adv_struct[:, 0:2, p+2]

        result = compute_angle(pt1, pt2, pt3)
        angles.append(result)

    combined = np.hstack(angles)
    fs = np.arange(frames)
    ts = fs * 1000.0 / 240.0
    final = np.hstack((fs, ts, combined))
    cols = ['Frame', 'Time (ms)', 'Ankle (deg)', 'Knee (deg)', 'Hip (deg)']
    fdf = pd.DataFrame(final, columns=cols)
    return fdf


def main():
    pass

if __name__ == '__main__':
    main()