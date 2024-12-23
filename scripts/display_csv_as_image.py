import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_images_from_csv(csv_file, num_images=10):
    """
    Display images from a 784/785-column CSV file to help determine ground truth.

    Parameters:
    - csv_file: Path to the CSV file containing the flattened image data.
    - num_images: Number of images to display (default: 10).
    """
    # Load the CSV file
    data = pd.read_csv(csv_file, header=None)
    
    # Handle 785 columns (785th column is the label)
    if data.shape[1] == 785:
        labels = data.iloc[:, 0].values  # Extract the ground truth labels
        pixels = data.iloc[:, 1:].values  # Extract pixel data
    elif data.shape[1] == 784:
        labels = None  # No labels available
        pixels = data.values
    else:
        raise ValueError(f"Unexpected number of columns: {data.shape[1]}. Expected 784 or 785.")

    # Display the specified number of images
    num_images = min(num_images, len(pixels))
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

    for i in range(num_images):
        # Reshape each row into a 28x28 image
        image = pixels[i].reshape(28, 28)
        
        # Plot the image
        plt.subplot(int(np.ceil(num_images / 5)), 5, i + 1)  # 5 images per row
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off the axes
        
        # Add title with ground truth label if available
        if labels is not None:
            plt.title(f"Label: {labels[i]}")
        else:
            plt.title(f"Image {i+1}")  # Use a generic title if no labels are present

    plt.tight_layout()
    plt.show()

# Main function to prompt user for file and display images
if __name__ == "__main__":
    csv_file = input("Enter the path to the CSV file: ").strip()
    try:
        num_images = int(input("Enter the number of images to display: ").strip())
        display_images_from_csv(csv_file, num_images)
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")