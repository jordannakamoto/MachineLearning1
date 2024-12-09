from part1 import part1
from part2 import part2
from part3 import part3


def main():

    # # Learning rate and epochs
    eta = 0.1
    epochs = 1000

    # Set file paths for training, validation, and testing
    train_files = [f"train{i}.csv" for i in range(10)]
    valid_files = [f"valid{i}.csv" for i in range(10)]
    test1 = "test.csv"
    test2and3 = "test1.csv"

    # Run Part 1 - Binary
    print("====================================================")
    print("Running Part 1: Binary Perceptron for Digits 7 and 9")
    print("====================================================")
    part1("train7.csv", "train9.csv", "valid7.csv", "valid9.csv", test1, epochs, eta)


    # Run Part 2
    print("====================================================")
    print("Running Part 2: Multiclass Perceptron for Digits 0-9")
    print("====================================================")
    part2(train_files, valid_files, test2and3, eta, epochs)

    # Run Part 3
    print("====================================================")
    print("Running Part 3: Neural Network Backpropagation for Digits 0-9")
    print("====================================================")
    part3(train_files, valid_files, test2and3,  eta=0.5, epochs=1000, target_accuracy=0.95)


if __name__ == "__main__":
    main()