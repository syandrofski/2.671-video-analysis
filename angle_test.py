import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy as dc
#import window_wrapper as ww

'''
# Define a function to compute the angle between three points
def angle_between_points(df):
    # Extract the X and Y coordinates of the three points
    x1, y1 = df.iloc[0]
    x2, y2 = df.iloc[1]
    x3, y3 = df.iloc[2]

    # Compute the vectors between the points
    v1 = np.array([x1-x2, y1-y2])
    v2 = np.array([x3-x2, y3-y2])

    # Compute the dot product and the length of the vectors
    dot_product = np.dot(v1, v2)
    length_v1 = np.linalg.norm(v1)
    length_v2 = np.linalg.norm(v2)

    # Compute the angle between the vectors
    angle = np.arccos(dot_product / (length_v1 * length_v2))

    # Return the angle in degrees
    return np.degrees(angle)

# Create a sample dataframe of three points
df = pd.DataFrame({
    'X': [0, 1, 2],
    'Y': [0, 1, 0]
})

# Compute the angle between the three points
angle = angle_between_points(df)

# Set the style and create the figure and axes
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(5,5))

# Plot the points using Seaborn
sns.scatterplot(x="X", y="Y", data=df, s=100, ax=ax)

# Add the angle as text to the plot
ax.text(1, 0.5, f"Angle: {angle:.2f}°", ha="center", va="center")

# Set the aspect ratio of the plot to 1:1
ax.set_aspect('equal')

# Show the plot
plt.show()
'''

def rename():
    # Directory path where the files are located
    directory = 'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\Final\\'

    # New file name prefix
    new_name_prefix = 'final_jump_'

    # Starting index for renaming
    start_index = 1

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Get the current file's full path
        file_path = os.path.join(directory, filename)

        # Check if the path corresponds to a file
        if os.path.isfile(file_path):
            # Generate the new file name with the desired prefix and sequential index
            new_filename = f'{new_name_prefix}{start_index}{os.path.splitext(filename)[1]}'
            #new_filename = 'final_jump_' + str(start_index) + '.mov'

            # Construct the new file's full path
            new_file_path = os.path.join(directory, new_filename)


            # Rename the file by moving it to the new path
            os.rename(file_path, new_file_path)

            # Increment the index for the next file
            start_index += 1


def interpolate(data):
    x = 0
    err = 1
    ct = 0
    rec = 0
    for j in range(data.shape[0]):
        if data[j, err] == 1:
            if ct == 0:
                rec = j - 1
            ct += 1
        elif ct > 0:
            x_rate = (data[j, x] - data[rec, x]) / float(j - rec)
            xb = data[rec, x]
            for k in range(rec + 1, j):
                data[k, x] = xb + (k - rec) * x_rate
                data[k, err] = 0
            ct = 0
            rec = 0
    return data

'''
a = np.zeros((10, 2))
for i in range(a.shape[0]):
    a[i, 0] = i
a[2, :] = np.array([0, 1])
a[4, :] = np.array([0, 1])
a[5, :] = np.array([0, 1])
print(np.transpose(a), '\n')
print(np.transpose(interpolate(a)))
'''

'''
data = np.load("C:\\Users\\spenc\\PycharmProjects\\2.671\\New Data Files\\Knee\\jump_1.npy")
print(data[20, :, 0], '\n')
data2 = np.load("C:\\Users\\spenc\\PycharmProjects\\2.671\\New Data Files\\Knee\\jump_5.npy")
print(data2[20, :, 0])
'''

#rename()