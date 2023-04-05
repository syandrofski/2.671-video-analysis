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

a = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0])
zeros = (np.array(np.where(np.flip(a)==0))+1)*-1
for z in zeros:
    print(a[z])