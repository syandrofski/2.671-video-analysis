import sys
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
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

def plot_together():
    # create an empty list to store all the dataframes
    dfs = []
    max_vels = []

    # specify the directory containing the csv files
    directory = "C:\\Users\\spenc\\PycharmProjects\\2.671\\New Data Files\\Knee\\"

    # loop through each file in the directory and read it into a dataframe
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, header=1, usecols=[2, 3], names=["Time", "Angle"])
            dfs.append(df)

    # create a figure and axes
    fig, ax = plt.subplots()
    f2, ax2 = plt.subplots()

    # loop through each dataframe and plot it on the same graph
    for i, df in enumerate(dfs):
        ax.plot(df["Time"].to_numpy(), np.gradient(df["Angle"].to_numpy())*240, label=f"File {i + 1}", alpha=0.65)
        max_vels.append(max(np.gradient(df["Angle"].to_numpy()*240)))
        ax2.plot(df["Time"].to_numpy()[-1], max_vels[-1], label=f"File {i + 1}", alpha=0.65)

    # add labels and a legend
    ax.set_xlabel("Time (on ground) [s]")
    ax.set_ylabel("Knee Angular Velocity [deg/s]")
    #ax.set_title("Angle vs. Time")
    #ax.legend()
    plt.savefig('agg_vel_graph')
    # show the plot
    plt.show()
    print(max_vels)
    return dfs

def plot_one_grad():
    # specify the directory containing the csv files
    fpath = "C:\\Users\\spenc\\PycharmProjects\\2.671\\Data Files\\jump5.csv"

    df = pd.read_csv(fpath, header=1, usecols=[2, 3], names=["Time", "Angle"])

    # create a figure and axes
    fig, ax = plt.subplots()

    ax.plot(df["Time"].to_numpy(), np.gradient(df["Angle"].to_numpy())*240, 'r-', color=(1.0, 165/255.0, 0))

    # add labels and a legend
    ax.set_xlabel("Time (on ground) [s]")
    ax.set_ylabel("Knee Angular Velocity [deg/s]")
    #ax.set_title("Angle vs. Time")
    #ax.legend()
    plt.savefig('single_vel_graph')
    # show the plot
    plt.show()

def plot_one_ang():
    # specify the directory containing the csv files
    fpath = "C:\\Users\\spenc\\PycharmProjects\\2.671\\New Data Files\\Knee\\jump_5.csv"

    df = pd.read_csv(fpath, header=1, usecols=[2, 3], names=["Time", "Angle"])

    # create a figure and axes
    fig, ax = plt.subplots()

    ax.plot(df["Time"].to_numpy(), df["Angle"], 'r-', color='blue')

    # add labels and a legend
    ax.set_xlabel("Time (on ground) [s]")
    ax.set_ylabel("Knee Angle [deg]")
    #ax.set_title("Angle vs. Time")
    #ax.legend()
    plt.savefig('single_ang_graph')
    # show the plot
    plt.show()

def get_mins():
    mins = []
    gcts = []
    dfs = plot_together()
    for df in dfs:
        mins.append(df['Angle'].min())
        gcts.append(df['Time'].iloc[-1])
    fig, ax  = plt.subplots()
    ax.scatter(mins, gcts)
    plt.show()
    print(mins)
    print(gcts)

def main():
    #plot_one_grad()
    plot_one_ang()
    #plot_together()

if __name__ == '__main__':
    main()