import numpy as np
import pandas as pd
from features import extract_features, normalize_features, get_global_min_max
import time

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability adjustment
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function
def cross_entropy_loss(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-8)) / predictions.shape[0]

def part3(train_files, valid_files, test_file, input_size=7, hidden_size=100, output_size=10, eta=0.01, epochs=1000, target_accuracy=0.95, decay_rate=0.0001, min_eta=0.000003):
    """
    Implements Neural Network Backpropagation for Digits 0-9.
    """
    print("Packing data...")
    # Load and preprocess training data
    train_data, train_labels = load_data(train_files)
    valid_data, valid_labels = load_data(valid_files)
    test_data, test_labels = load_data([test_file])

    # Compute global min and max for normalization
    global_min, global_max = get_global_min_max(train_data, valid_data, test_data)

    # Normalize features using global min/max
    train_data = normalize_features(train_data, global_min, global_max)
    valid_data = normalize_features(valid_data, global_min, global_max)
    test_data = normalize_features(test_data, global_min, global_max)

    # Train the neural network
    best_W1, best_b1, best_W2, best_b2, best_accuracy, min_error, min_error_epoch = train_neural_network(
        train_data, train_labels, valid_data, valid_labels, input_size, hidden_size, output_size, eta, epochs, target_accuracy
    )

    print("Training Done!")
    print("------------------------")
    time.sleep(2)
    print(f"Minimum validation error: {min_error} @ epoch {min_error_epoch}")
    time.sleep(2)
    print("Best weights (W1, b1, W2, b2):", best_W1, best_b1, best_W2, best_b2)

    # Test the neural network
    test_Z1 = np.dot(test_data, best_W1) + best_b1
    test_A1 = relu(test_Z1)
    test_Z2 = np.dot(test_A1, best_W2) + best_b2
    test_A2 = softmax(test_Z2)
    test_predictions = np.argmax(test_A2, axis=1)

    # Calculate accuracy on the test set
    test_accuracy = np.mean(test_predictions == test_labels)
    # Print accuracy per class (0-9)
    for class_label in range(10):
        # Get indices for the current class
        class_indices = np.where(test_labels == class_label)[0]
        # Compute accuracy for this class
        class_accuracy = np.mean(test_predictions[class_indices] == class_label)
        print(f"Accuracy for class {class_label}: {class_accuracy:.4f}")
    print("--------------------------------")
    print(f"Total Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save predictions
    with open("output_part3.txt", "w") as f:
        f.write(" ".join(map(str, test_predictions)))
    time.sleep(2)
    
    print("Predictions saved to output_part3.txt")
    print("--------------------------------")


def load_data(files):
    """
    Load and preprocess data from the given CSV files.
    Returns the features and labels.
    """
    data_list = []
    labels_list = []

    for file in files:
        data = pd.read_csv(file, header=None)
        labels = data.iloc[:, 0].values  # Labels are in the first column
        features = extract_features(data)
        data_list.append(features)
        labels_list.append(labels)
    # Combine all files' data into single arrays
    data_combined = np.vstack(data_list)
    labels_combined = np.hstack(labels_list)

    return data_combined, labels_combined

def train_neural_network(train_data, train_labels, valid_data, valid_labels, input_size, hidden_size, output_size, eta=0.1, epochs=1000, target_accuracy=0.95, decay_rate=0.0001, min_eta=0.000003):
    # Initialize weights and biases
    W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))  # Input to hidden layer
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(-0.1, 0.1, (hidden_size, output_size))  # Hidden to output layer
    b2 = np.zeros((1, output_size))

    best_W1, best_b1, best_W2, best_b2 = W1, b1, W2, b2
    best_accuracy = 0
    min_error_epoch = 0
    min_error = 0

    # One-hot encode labels
    train_labels_onehot = np.eye(output_size)[train_labels]
    valid_labels_onehot = np.eye(output_size)[valid_labels]

    for epoch in range(epochs):
        # Forward pass
        Z1 = np.dot(train_data, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)

        # Compute loss
        loss = cross_entropy_loss(A2, train_labels_onehot)

        # Backward pass
        dZ2 = A2 - train_labels_onehot
        dW2 = np.dot(A1.T, dZ2) / train_data.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / train_data.shape[0]

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(train_data.T, dZ1) / train_data.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / train_data.shape[0]

        # Update weights and biases
        W2 -= eta * dW2
        b2 -= eta * db2
        W1 -= eta * dW1
        b1 -= eta * db1

        # Validate
        valid_Z1 = np.dot(valid_data, W1) + b1
        valid_A1 = relu(valid_Z1)
        valid_Z2 = np.dot(valid_A1, W2) + b2
        valid_A2 = softmax(valid_Z2)

        valid_predictions = np.argmax(valid_A2, axis=1)
        accuracy = np.mean(valid_predictions == valid_labels)
        error_rate = np.mean(valid_predictions != valid_labels)

        # Decay learning rate after each epoch
        eta = max(eta / (1 + decay_rate), min_eta)

        # Save best weights
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
            min_error = error_rate
            min_error_epoch = epoch + 1

        # Early stopping
        if accuracy >= target_accuracy:
            print(f"Target accuracy of {target_accuracy*100}% reached at epoch {epoch + 1}")
            break

        # Print validation error every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Validation Error: {error_rate:.4f}")

    return best_W1, best_b1, best_W2, best_b2, best_accuracy, min_error, min_error_epoch