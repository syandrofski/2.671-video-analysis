import os
from os.path import isfile
import csv
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from math import pi
from curvefitting import cftool
import scipy.stats as stats

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


def summary(_dir, overwrite=False):
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
        if filename.endswith('.csv') and filename.startswith('opt_jump_'):
            run_num += 1
            print(filename)
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                data = np.array([row[3:6] for row in reader], dtype=float)  # Read Ankle, Knee, Hip columns

            # Compute the gradient over time for each joint
            gradients = np.apply_along_axis(compute_gradient, axis=0, arr=data)

            # Find minimum value and frame number after the halfway point for each joint
            for (i, (joint, joint_data_dict)) in enumerate(joint_data.items()):
                joint_data_dict['run'].append(run_num)
                joint_data_dict['total_frames'].append(data.shape[0])

                min_value, min_frame = find_minimum(data[:, i])
                joint_data_dict['min_value'].append(min_value)
                print(min_value)
                joint_data_dict['min_frame'].append(min_frame)

                max_grad, max_grad_frame = find_max_gradient(gradients[:, i])
                joint_data_dict['max_gradient'].append(max_grad)
                joint_data_dict['max_grad_frame'].append(max_grad_frame)

    # Export the computed values to a CSV file
    csv_filename = _dir + 'joint_data.csv'
    if not isfile(csv_filename) or overwrite:
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
                writer.writerows(zip(run, total_frames,[joint] * len(min_values), min_values, min_frames, max_gradients, max_grad_frames))
    return joint_data


def plot_all(_dir, single=0, gradient=False):
    # Directory path where the data files are located
    directory = _dir

    # Prefix to filter the data files
    file_prefix = 'opt_jump_'

    # Colors and opacity for plotting
    colors = {'Ankle': 'red', 'Knee': 'blue', 'Hip': 'orange'}
    if single < 1:
        opacity = {'Ankle': 0.35, 'Knee': 0.3, 'Hip': 0.6}
    else:
        opacity = {'Ankle': 1, 'Knee': 1, 'Hip': 1}

    ct = 0

    time = []
    ankle_angle = []
    knee_angle = []
    hip_angle = []

    # Create subplots for each joint angle
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5.4, 4.8), sharex='all')

    # Iterate over each file in the directory
    for i, filename in enumerate(os.listdir(directory)):
        if filename.startswith(file_prefix) and filename.endswith('.csv'):
            ct += 1
            if single < 1:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # Skip header row

                    # Lists to store the time and angle measurements for each joint
                    time = []
                    ankle_angle = []
                    knee_angle = []
                    hip_angle = []

                    for row in reader:
                        time.append(float(row[2]))  # Time (ms)
                        ankle_angle.append(float(row[3]))  # Ankle (deg)
                        knee_angle.append(float(row[4]))  # Knee (deg)
                        hip_angle.append(float(row[5]))  # Hip (deg)
            else:
                if single == ct:
                    filepath = os.path.join(directory, filename)
                    print(filename)
                    with open(filepath, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        next(reader)  # Skip header row

                        # Lists to store the time and angle measurements for each joint
                        time = []
                        ankle_angle = []
                        knee_angle = []
                        hip_angle = []

                        for row in reader:
                            time.append(float(row[2]))  # Time (ms)
                            ankle_angle.append(float(row[3]))  # Ankle (deg)
                            knee_angle.append(float(row[4]))  # Knee (deg)
                            hip_angle.append(float(row[5]))  # Hip (deg)

                    if gradient:
                        ankle_angle, knee_angle, hip_angle = np.gradient(ankle_angle), np.gradient(
                            knee_angle), np.gradient(hip_angle)
                    # Plot ankle angles
                    ax1.plot(time, ankle_angle, color=colors['Ankle'], alpha=opacity['Ankle'])

                    # Plot knee angles
                    ax2.plot(time, knee_angle, color=colors['Knee'], alpha=opacity['Knee'])

                    # Plot hip angles
                    ax3.plot(time, hip_angle, color=colors['Hip'], alpha=opacity['Hip'])

            if single < 1:
                if gradient:
                    ankle_angle, knee_angle, hip_angle = np.gradient(ankle_angle), np.gradient(knee_angle), np.gradient(hip_angle)
                # Plot ankle angles
                ax1.plot(time, ankle_angle, color=colors['Ankle'], alpha=opacity['Ankle'])

                # Plot knee angles
                ax2.plot(time, knee_angle, color=colors['Knee'], alpha=opacity['Knee'])

                # Plot hip angles
                ax3.plot(time, hip_angle, color=colors['Hip'], alpha=opacity['Hip'])

    # Set plot title and labels
    plt.xlabel('Time (ms)', fontsize=14)
    if gradient:
        #plt.title('Angular Velocity Measurements Over Time')
        ax2.set_ylabel('Angular Velocity (deg/ms)', fontsize=14)
    else:
        #plt.title('Angle Measurements Over Time')
        ax2.set_ylabel('Angle (deg)', fontsize=14)

    # Set legend
    legend_labels_ax1 = ['Ankle']
    legend_labels_ax2 = ['Knee']
    legend_labels_ax3 = ['Hip']
    ax1.legend(legend_labels_ax1)
    ax2.legend(legend_labels_ax2)
    ax3.legend(legend_labels_ax3)

    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_scatter(_dir, gradient=False):
    # Read the CSV file
    data = pd.read_csv(_dir)

    # Group the data by joint
    grouped_data = data.groupby('Joint')

    # Define colors for each joint
    colors = {'Ankle': 'red', 'Knee': 'blue', 'Hip': 'orange'}

    # Create a scatterplot for each joint
    for joint, group in grouped_data:
        # Get the Min Value and Total Frames values for the current joint
        min_value = group['Min Value']
        total_frames = group['Total Frames']
        times = total_frames.to_numpy() * 1000.0 / 240.0

        # Create a new figure and axis for each joint
        fig, ax = plt.subplots()

        # Plot the Min Value by Total Frames pairs with the specified color
        ax.scatter(times, min_value, color=colors[joint])

        # Set plot title and labels
        #ax.set_title(f'{joint} - Min Value by Total Frames')
        if joint == 'Hip' and 'Proc2' in _dir:
            plt.yticks([125.2, 125.4, 125.6, 125.8, 126.0, 126.2, 126.4, 126.6, 126.8])
        ax.set_xlabel('Time (ms)', fontsize = 14)
        ax.set_ylabel(f'{joint} Minimum Angle (deg)', fontsize = 14)

        # Display the plot for the current joint
        plt.show()

def plot_hist(_dir):
    # Read the CSV file
    data = pd.read_csv(_dir)

    # Group the data by joint
    grouped_data = data.groupby('Joint')
    colors = {'Ankle': 'red', 'Knee': 'blue', 'Hip': 'orange'}

    result = {}

    # Create a histogram for each joint
    for joint, group in grouped_data:
        # Get the angle data for the current joint
        angles = group['Min Value']

        # Create a new figure and axis for each joint
        fig, ax = plt.subplots()

        # Plot the histogram of angle data
        n, bins, patches = ax.hist(angles, bins=8, color=colors[joint], edgecolor='black')

        tx = ax.get_xlim()
        ty = ax.get_ylim()

        # Fit a normal distribution to the angle data
        mu, sigma = stats.norm.fit(angles)
        result[joint] = {'mean': mu, 'std': sigma}

        # Plot the fitted normal distribution curve
        #curve_range = range(int(min(angles)), int(max(angles)) + 1)
        curve_range = np.linspace(int(min(angles)-10), int(max(angles)+10), 100)
        curve = stats.norm.pdf(curve_range, mu, sigma) * len(angles) * (bins[1] - bins[0])
        ax.plot(curve_range, curve, 'k-', linewidth=2)

        ax.set_xlim(tx)
        ax.set_ylim(ty)

        '''
        # Plot the histogram of angle data
        ax.hist(angles, bins=8, color=colors[joint], edgecolor='black')
        '''

        # Set plot title and labels
        #ax.set_title(f'{joint} - Angle Histogram')
        ax.set_xlabel(f'{joint} Angle (deg)', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)

        # Display the plot for the current joint
        plt.show()

    return result

def unit_test(n, _dir):
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
    if n == 4:
        joint_data = summary('C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\')
        knee_data = joint_data['Knee']
        min_vals = knee_data['min_value']
        print(type(min_vals))
        print(min_vals)
        frames = knee_data['total_frames']
        things = {'Knee_Min_Vals': np.array(min_vals), 'Frames': np.array(frames)}
        cftool(things)
    if n == 5:
        plot_all(_dir, single=0, gradient=True)
    else:
        print('Please enter a valid unit test code.')

def main():
    #joint_data = summary('C:\\Users\\spenc\\PycharmProjects\\2.671\\Final Data Files\\')
    #plot_all('C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\', single=0, gradient=True)
    #plot_scatter('C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\joint_data.csv')
    res = plot_hist('C:\\Users\\spenc\\PycharmProjects\\2.671\\Final Data Files\\joint_data_pruned.csv')
    print(res)

if __name__ == '__main__':
    main()