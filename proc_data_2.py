import os
import csv
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


def summary(_dir):
    # Function to compute the gradient over time
    def compute_gradient(data):
        return np.gradient(data)

    # Function to find the minimum value and frame number after the halfway point
    def find_minimum(data):
        halfway_idx = len(data) // 2
        min_value = np.min(data[halfway_idx:])
        min_frame = np.argmin(data[halfway_idx:]) + halfway_idx
        return min_value, min_frame

    # Function to find the maximum gradient value and frame number after the halfway point
    def find_max_gradient(gradient):
        halfway_idx = len(gradient) // 2
        max_grad = np.max(gradient[halfway_idx:])
        max_frame = np.argmax(gradient[halfway_idx:]) + halfway_idx
        return max_grad, max_frame

    # Directory path where the CSV files are located
    directory = _dir

    # Initialize a dictionary to store the computed values for each joint
    joint_data = {
        'Ankle': {'run': [], 'total_frames': [], 'min_value': [], 'min_frame': [], 'max_gradient': [], 'max_grad_frame': []},
        'Knee': {'run': [], 'total_frames': [], 'min_value': [], 'min_frame': [], 'max_gradient': [], 'max_grad_frame': []},
        'Hip': {'run': [], 'total_frames': [], 'min_value': [], 'min_frame': [], 'max_gradient': [], 'max_grad_frame': []}
    }
    run_num = 0
    # Iterate over each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            run_num += 1
            print(filename)
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                data = np.array([row[3:6] for row in reader], dtype=float)  # Read Ankle, Knee, Hip columns
                print(data)

            # Compute the gradient over time for each joint
            gradients = np.apply_along_axis(compute_gradient, axis=0, arr=data)

            # Find minimum value and frame number after the halfway point for each joint
            for (i, (joint, joint_data_dict)) in enumerate(joint_data.items()):
                min_value, min_frame = find_minimum(data[:, i])
                joint_data_dict['run'].append(run_num)
                joint_data_dict['total_frames'].append(data.shape[0])
                joint_data_dict['min_value'].append(min_value)
                joint_data_dict['min_frame'].append(min_frame)

            # Find maximum gradient value and frame number after the halfway point for each joint
            for (i, (joint, joint_data_dict)) in enumerate(joint_data.items()):
                joint_data_dict['run'].append(run_num)
                joint_data_dict['total_frames'].append(data.shape[0])
                max_grad, max_grad_frame = find_max_gradient(gradients[:, i])
                joint_data_dict['max_gradient'].append(max_grad)
                joint_data_dict['max_grad_frame'].append(max_grad_frame)

    # Export the computed values to a CSV file
    csv_filename = 'joint_data.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Run', 'Total Frames', 'Joint', 'Min Value', 'Min Frame', 'Max Gradient', 'Max Grad Frame'])
        for joint, joint_data_dict in joint_data.items():
            run = joint_data_dict['run']
            total_frames = joint_data_dict['total_frames']
            min_values = joint_data_dict['min_value']
            min_frames = joint_data_dict['min_frame']
            max_gradients = joint_data_dict['max_gradient']
            max_grad_frames = joint_data_dict['max_grad_frame']
            writer.writerows(zip(run, total_frames,[joint] * len(min_values), min_values, min_frames, max_gradients, max_grad_frames, total_frames))


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
    elif n == 4:
        summary('C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\')
    else:
        print('Please enter a valid unit test code.')

def main():
    unit_test(4)

if __name__ == '__main__':
    main()