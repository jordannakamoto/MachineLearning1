import numpy as np
import pandas as pd
from collections import deque

"""
# FEATURES

Methods:
Normalize Features
  - normalize extracted features (0.0-1.0) - arg: global min/max
Get Global Min Max
    - for normalization across datasets
Extract Features
  - methods to extract 7 features from a 785 column csv

The 7 features from each MNIST image:
1. Density
2. Symmetry
3/4. Max/Avg Horizontal Intersections
5/6. Max/Avg Horizontal Intersections
7. Connected components (circles)
"""

# Normalize features between two global points to between 0 and 1
def normalize_features(data, global_min, global_max):
    return (data - global_min) / (global_max - global_min)

# Get global min max for normalization given train, valid, test
def get_global_min_max(train_data, valid_data, test_data):
    """
    Calculate global min and max values for normalization
    across training, validation, and test datasets.
    """
    # Combine all features into one dataset for min/max calculation
    all_data = np.vstack((train_data, valid_data, test_data))
    
    # Compute the global min and max for each feature
    global_min = all_data.min(axis=0)
    global_max = all_data.max(axis=0)
    
    return global_min, global_max

# Extract Features
def extract_features(data):
    num_samples = data.shape[0]

    # Assume the first column is the digit label (0-9)
    # or if there are 784 columns like in part1, that there is no label column
    # Differentiate between arrays and Pandas DataFrame
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 785: 
            pixels = data.iloc[:, 1:].values
        else:
            pixels = data.iloc[:, :].values
    elif isinstance(data, np.ndarray):
        if data.shape[1] == 785:
            pixels = data[:, 1:]
        else:
            pixels = data
    else:
        raise ValueError("Input data must be a Pandas DataFrame or a NumPy array.")
    

    features = np.zeros((num_samples, 7))

    for i, row in enumerate(pixels):
        image = row.reshape(28, 28)  # Reshape flattened image into 28x28
        binary_image = (image > 128).astype(int)

        # Feature 1: Density
        # average gray scale value of all the pixels in the image
        density =  np.sum(image) / (28 * 28)

        # Feature 2: Symmetry (difference between top and bottom halves)
        # average gray scale of the image obtained by the bitwise XOR (⊕) of each pixel with its corresponding vertically reflected image.
        # Convert image to binary (grayscale thresholding)
        # gray scale values above (below) 128
        reflected_image = np.flipud(binary_image)
        # XOR operation between the original and reflected images
        xor_image = np.bitwise_xor(binary_image, reflected_image)
        # Calculate the density of the XOR image
        symmetry = np.sum(xor_image) / (28 * 28)

        # Feature 3 & 4: Calculate horizontal intersections
        horizontal_intersections = [
            np.sum(row[:-1] != row[1:]) for row in binary_image
        ]
        max_horizontal_intersections = np.max(horizontal_intersections)
        avg_horizontal_intersections = np.mean(horizontal_intersections)

        # Feature 5 & 6: Vertical INtersections
        vertical_intersections = [
            np.sum(col[:-1] != col[1:]) for col in binary_image.T
        ]
        max_vertical_intersections = np.max(vertical_intersections)
        avg_vertical_intersections = np.mean(vertical_intersections)

        # # Feature 7: A seventh feature...?
        # # horizontal symmetry
        # reflected_image = np.fliplr(binary_image)
        # # XOR operation between the original and reflected images
        # xor_image = np.bitwise_xor(binary_image, reflected_image)
        # # Calculate the density of the XOR image
        # symmetryh = np.sum(xor_image) / (28 * 28)

        # Feature 8: Count Enclosed Regions (Loops)
        loop_count = count_connected_components(1 - binary_image) - 1  # Subtract 1 for the background


        # Store features
        features[i] = [
            density,
            symmetry,
            max_horizontal_intersections / 28,
            avg_horizontal_intersections / 28,
            max_vertical_intersections / 28,
            avg_vertical_intersections / 28,
            loop_count,
            ]

    return features

# Helper function to count connected components in a binary image
def count_connected_components(binary_image):
    """
    Counts the number of connected components (loops) in the binary image.
    - A connected component is a group of 1s surrounded by 0s.
    """
    visited = np.zeros_like(binary_image, dtype=bool)
    rows, cols = binary_image.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def bfs(start_row, start_col):
        queue = deque([(start_row, start_col)])
        visited[start_row, start_col] = True

        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and binary_image[nr, nc] == 1:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # Count connected components
    num_components = 0
    for r in range(rows):
        for c in range(cols):
            if binary_image[r, c] == 1 and not visited[r, c]:
                bfs(r, c)
                num_components += 1

    return num_components