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

    angle_rad = pi - np.arccos(cos_ang)
    angle_deg = np.degrees(angle_rad)

    return np.reshape(angle_deg, (angle_deg.size, 1))
    #return np.reshape(angle_rad, (angle_rad.size, 1))

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
    fs = np.reshape(fs, (fs.size, 1))
    ts = fs * 1000.0 / 240.0
    final = np.hstack((fs, ts, combined))
    cols = ['Frame', 'Time (ms)', 'Ankle (deg)', 'Knee (deg)', 'Hip (deg)']
    fdf = pd.DataFrame(final, columns=cols)
    return fdf

def unit_test(n):
    tgts = np.array([0, 1, 2])
    tgts += 1

    if n == 1:
        tp1 = np.array([[1, 1], [2, 1]])
        tp2 = np.array([[0, 0], [1, 0]])
        tp3 = np.array([[1, 0], [2, 0]])
        print(compute_angle(tp1, tp2, tp3))
    elif n == 2:
        data = np.load("C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\opt_jump_1.npy")
        print('Final Points:\n', data[0, 0:2, 0:3])
        pt1 = data[:6, 0:2, tgts[0]]
        pt2 = data[:6, 0:2, tgts[1]]
        pt3 = data[:6, 0:2, tgts[2]]
        print('Point Data:\n', np.hstack((pt1, pt2, pt3))[0, :])
        print('Angle Output:\n', compute_angle(pt1, pt2, pt3)[0], '\n')
        plot_points = np.vstack((pt1[0, :], pt2[0, :], pt3[0, :]))
        #print(plot_points)
        plt.scatter(plot_points[:, 0], -1*plot_points[:, 1])#, c=['blue', 'red', 'yellow'])
        plt.gca().set_aspect('equal')
        #plt.show()
    elif n == 3:
        data = np.load("C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\opt_jump_1.npy")
        print('Final Points:\n', data[-1, 0:2, 0:3])
        pt1 = data[-5:, 0:2, tgts[0]]
        pt2 = data[-5:, 0:2, tgts[1]]
        pt3 = data[-5:, 0:2, tgts[2]]
        print('Point Data:\n', np.hstack((pt1, pt2, pt3))[0, :])
        print('Angle Output:\n', compute_angle(pt1, pt2, pt3)[0])
        plot_points = np.vstack((pt1[-1, :], pt2[-1, :], pt3[-1, :]))
        #print(plot_points)
        plt.scatter(plot_points[:, 0], -1*plot_points[:, 1], c='red')#, c=['blue', 'red', 'yellow'])
        plt.gca().set_aspect('equal')
        plt.show()
    else:
        print('Please enter a valid unit test code.')

def main():
    unit_test(2)
    unit_test(3)

if __name__ == '__main__':
    main()