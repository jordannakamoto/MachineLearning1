import numpy as np
import pandas as pd
from features import extract_features, normalize_features, get_global_min_max
import time


def part2(train_files, valid_files, test_file, eta=0.01, epochs=1000, decay_rate=0.0001, min_eta=0.000003):
    """
    Implements Part 2: Multiclass perceptron training for digits 0-9.
    """
    print("Packing data...")
    # Load and preprocess training data
    train_data = []
    train_labels = []
    for label, train_file in enumerate(train_files):
        data = pd.read_csv(train_file, header=None)
        labels = data.iloc[:, 0].values  # Extract the labels from the first column
        features = extract_features(data.iloc[:, 1:])  # Extract features (drop the first column for labels)
        train_data.append(features)
        train_labels.append(labels)
    
    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)

    # Load and preprocess validation data
    valid_data = []
    valid_labels = []
    for label, valid_file in enumerate(valid_files):
        data = pd.read_csv(valid_file, header=None)
        labels = data.iloc[:, 0].values  # Extract the labels from the first column
        features = extract_features(data.iloc[:, 1:])  # Extract features (drop the first column for labels)
        valid_data.append(features)
        valid_labels.append(labels)

    valid_data = np.vstack(valid_data)
    valid_labels = np.hstack(valid_labels)

    # Load and preprocess test data
    test_data = pd.read_csv(test_file, header=None)
    test_labels = test_data.iloc[:, 0].values  # Extract the labels from the first column
    test_features = extract_features(test_data.iloc[:, 1:])  # Extract features (drop the first column for labels)

    # Compute global min and max for normalization
    global_min, global_max = get_global_min_max(train_data, valid_data, test_features)

    # Normalize features using global min/max
    train_data = normalize_features(train_data, global_min, global_max)
    valid_data = normalize_features(valid_data, global_min, global_max)
    test_features = normalize_features(test_features, global_min, global_max)

    # Add bias term (column of 1s) to the test data
    test_features = np.hstack((test_features, np.ones((test_features.shape[0], 1))))  # Add bias term to test data

    # Train multiclass perceptron
    best_weights, min_error, min_error_epoch = train_multiclass_perceptron(train_data, train_labels, valid_data, valid_labels, num_classes=10, eta=eta, epochs=epochs, decay_rate=decay_rate, min_eta=min_eta)

    print("Training Done!")
    print("------------------------")
    time.sleep(2)
    print(f"Minimum validation error: {min_error} @ epoch {min_error_epoch}")
    time.sleep(2)
    print("Best weights (7 inputs + bias term):", best_weights)

    # Test the perceptron on the test data
    test_predictions = test_multiclass_perceptron(test_features, test_labels, best_weights)
    # Reporting handled by this function ^

    # Save predictions to output file
    with open("output_part2.txt", "w") as f:
        f.write(" ".join(map(str, test_predictions)))
    time.sleep(2)
    print("Predictions saved to output_part2.txt")

# Multiclass Perceptron Training with Learning Rate Decay
def train_multiclass_perceptron(train_data, train_labels, valid_data, valid_labels, num_classes=10, eta=0.01, epochs=1000, decay_rate=0.0001, min_eta=0.000003):
    """
    Trains a multiclass perceptron using the training data and returns the best weights.
    Also implements learning rate decay with a minimum learning rate threshold.
    """
    num_features = 7
    weights = np.random.uniform(-0.01, 0.01, (num_classes, num_features + 1))  # Adding one more for the bias term
    best_weights = weights.copy()
    min_error = float('inf')
    min_error_epoch = 0

    # Add bias term (column of 1s) to the training and validation data
    train_data = np.hstack((train_data, np.ones((train_data.shape[0], 1))))  # Add bias term to training data
    valid_data = np.hstack((valid_data, np.ones((valid_data.shape[0], 1))))  # Add bias term to validation data

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        for i, x in enumerate(train_data):
            true_label = train_labels[i]
            scores = np.dot(weights, x)
            predicted_label = np.argmax(scores)

            if predicted_label != true_label:
                # Update weights for true and predicted classes
                weights[true_label] += eta * x
                weights[predicted_label] -= eta * x

        # Decay learning rate after each epoch
        eta = max(eta / (1 + decay_rate), min_eta)

        # Validate on validation data
        valid_predictions = np.argmax(np.dot(valid_data, weights.T), axis=1)
        error_rate = np.mean(valid_predictions != valid_labels)

        # Print validation error every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Validation Error: {error_rate:.4f}")

        # Store the best weights if the error rate improves
        if error_rate < min_error:
            min_error = error_rate
            best_weights = weights.copy()
            min_error_epoch = epoch + 1

    return best_weights, min_error, min_error_epoch

def test_multiclass_perceptron(test_features, test_labels, best_weights):
    """
    Tests the multiclass perceptron on the test set and computes accuracy.
    Prints overall accuracy and accuracy per class (0-9).
    """
    # Predict using the learned best weights
    test_predictions = np.argmax(np.dot(test_features, best_weights.T), axis=1)

    # Compute overall accuracy by comparing predicted labels with true labels
    overall_accuracy = np.mean(test_predictions == test_labels)
    
    # Print overall accuracy
    print(f"Overall Test Accuracy: {overall_accuracy:.4f}")

    # Print accuracy per class (0-9)
    for class_label in range(10):
        # Get indices for the current class
        class_indices = np.where(test_labels == class_label)[0]
        # Compute accuracy for this class
        class_accuracy = np.mean(test_predictions[class_indices] == class_label)
        print(f"Accuracy for class {class_label}: {class_accuracy:.4f}")

    return test_predictions