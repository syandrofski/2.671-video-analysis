import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from math import pi

def compute_angle(pts1, pts2, pts3):
    vec12 = pts2 - pts1
    vec23 = pts3 - pts2

    dot_product = np.sum(vec12 * vec23, axis=1)

    mag_12 = np.linalg.norm(vec12, axis=1)
    mag_23 = np.linalg.norm(vec23, axis=1)

    cos_ang = dot_product / (mag_12 * mag_23)

    angle_rad = np.arccos(cos_ang)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

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

def unit_test(n):
    if n == 1:
        tp1 = np.array([[1, 1], [2, 1]])
        tp2 = np.array([[0, 0], [1, 0]])
        tp3 = np.array([[1, 0], [2, 0]])
        print(compute_angle(tp1, tp2, tp3))

def main():
    unit_test(1)

if __name__ == '__main__':
    main()