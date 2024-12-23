import numpy as np
import pandas as pd
import time

from features import extract_features,normalize_features

"""
Usage:
This script trains and tests a perceptron model for classifying digits `7` and `9` using extracted features.

Functions:
- part1(train7, train9, valid7, valid9, test_file, epochs=1000, eta=0.1): 
   Loads data, extracts features, trains the perceptron, and tests it.
- train_perceptron(train_data, train_labels, valid_data, valid_labels, epochs=1000, eta=0.1): 
   Trains the perceptron.
- test_perceptron(test_features, weights): 
   Tests the perceptron and prints accuracy.
"""


# Main function for Part 1
def part1(train7, train9, valid7, valid9, test_file, epochs=1000, eta=0.1, decay_rate = .0001, min_eta = .000003):
    # Load data from file
    train7_data = pd.read_csv(train7, header=None)
    train9_data = pd.read_csv(train9, header=None)
    valid7_data = pd.read_csv(valid7, header=None)
    valid9_data = pd.read_csv(valid9, header=None)
    test_data = pd.read_csv(test_file, header=None)

    # Extract features
    train7_features = extract_features(train7_data)
    train9_features = extract_features(train9_data)
    valid7_features = extract_features(valid7_data)
    valid9_features = extract_features(valid9_data)
    test_features = extract_features(test_data)

    # Combine all features to compute global min and max for normalization
    all_features = np.vstack((train7_features, train9_features, valid7_features, valid9_features, test_features))
    global_min = all_features.min(axis=0)
    global_max = all_features.max(axis=0)

    # Normalize features using global min and max
    train7_features = normalize_features(train7_features, global_min, global_max)
    train9_features = normalize_features(train9_features, global_min, global_max)
    valid7_features = normalize_features(valid7_features, global_min, global_max)
    valid9_features = normalize_features(valid9_features, global_min, global_max)
    test_features = normalize_features(test_features, global_min, global_max)

    # Merge 7,9 data and define class labels
    train_data = np.vstack((train7_features, train9_features))
    valid_data = np.vstack((valid7_features, valid9_features))
    # DEFINE 7 as the positive class
    train_labels = np.array([1] * train7_features.shape[0] + [-1] * train9_features.shape[0])
    valid_labels = np.array([1] * valid7_features.shape[0] + [-1] * valid9_features.shape[0])

    # Train perceptron
    best_weights, min_error, min_error_epoch = train_perceptron(train_data, train_labels, valid_data, valid_labels, epochs, eta, decay_rate, min_eta)

    # Report
    print("Training Done!")
    print("------------------------")
    time.sleep(2)
    print(f"Minimum validation error: {min_error} @ epoch {min_error_epoch}")
    time.sleep(2)
    print("Best weights (7 inputs + bias term):", best_weights)
    time.sleep(2)

    # Test perceptron
    test_predictions = test_perceptron(test_features, best_weights)

    # Write predictions to output file
    with open("output1_part1.txt", "w") as f:
        f.write(" ".join(map(str, test_predictions)))
    time.sleep(2)
    print("Predictions saved to output_part1.txt")
    time.sleep(3)

    



# Perceptron training algorithm
def train_perceptron(train_data, train_labels, valid_data, valid_labels, epochs=1000, eta=0.1, decay_rate = .0001, min_eta = .000003):
    # Add bias term (column of 1s) to the data
    train_data = np.hstack((train_data, np.ones((train_data.shape[0], 1))))
    valid_data = np.hstack((valid_data, np.ones((valid_data.shape[0], 1))))

    # Initialize weights including the bias term
    weights = np.random.uniform(-0.1, 0.1, train_data.shape[1])
    best_weights = weights.copy()
    min_error = float('inf')
    min_error_epoch = 0

    # decay_rate = 0.0001  # Define a small learning decay rate
    # min_eta = .000003

    for epoch in range(epochs):
        for i, x in enumerate(train_data):
            y_pred = np.sign(np.dot(weights, x))
            y_true = train_labels[i]
            if y_pred != y_true:
                weights += eta * y_true * x
        
        # Decay learning rate after each epoch
        eta = max(eta / (1 + epoch * decay_rate), min_eta)

        # Evaluate on validation data
        valid_predictions = np.sign(np.dot(valid_data, weights))
        error_rate = np.mean(valid_predictions != valid_labels)

        # Print validation error every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Validation Error: {error_rate:.4f}")

        if error_rate < min_error:
            min_error = error_rate
            best_weights = weights.copy()
            min_error_epoch = epoch + 1  # Store the epoch where min error occurs for report


    return best_weights, min_error, min_error_epoch

# Test perceptron on a test dataset
# ASSUMES format of test.csv
def test_perceptron(test_features, weights, num_samples=100):
    # Add bias term (column of 1s) to the test data
    test_features = np.hstack((test_features, np.ones((test_features.shape[0], 1))))  # Add bias term
    
    # Predict using the learned best weights arg
    test_predictions = np.sign(np.dot(test_features, weights))
    
    # Convert predictions from binary -1/1 to 7/9 for reporting
    predicted_labels = np.where(test_predictions > 0, 7, 9)
    
    # Create ground truth based on the given ranges (0-9)
    # of test.csv
    # Every 10 samples correspond to a different label (0-9)
    true_labels = np.repeat(np.arange(10), num_samples // 10)
    
    # Compute overall accuracy (all classes)
    overall_accuracy = np.mean(predicted_labels == true_labels)
    
    # Identify indices where the true labels are 7 or 9
    relevant_indices = np.isin(true_labels, [7, 9])
    
    # Compute accuracy for just the 7s and 9s
    relevant_predictions = predicted_labels[relevant_indices]
    relevant_true_labels = true_labels[relevant_indices]
    relevant_accuracy = np.mean(relevant_predictions == relevant_true_labels)
    
    print("------------------------")
    print("Test Results on test.csv")
    # Print both the overall accuracy and the 7/9 accuracy
    print(f"Test Accuracy on All Digits in 'test.csv': {overall_accuracy:.4f}")
    print(f"Accuracy for 7s and 9s: {relevant_accuracy:.4f}")
    print(f"(assuming every 10 rows is a digit in test.csv where 71-80 are 7's)")

    return predicted_labels