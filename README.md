# **MNIST Dataset Classifiers: 7 Features**
Famous MNIST Dataset: takes an image of a handwritten digit represented as a CSV and predicts (0-9).

**Part 1: Single Perceptron Classifier**  
A binary perceptron to classify digits **7** and **9** using the pocket algorithm. Outputs optimal weights, error rates, and predictions for the test data.

**Part 2: Multi-Class Perceptron**  
Extends the perceptron to classify all **ten digits (0–9)**. Outputs error rates and predictions for the test data.

**Part 3: Neural Network (Backpropagation)**  
Implements a neural network with:  
\- **One hidden layer** containing **100 neurons**.  
\- **ReLU activation** for the hidden layer.  
\- **Softmax activation** for the output layer to perform multi-class classification.  
   Training stops when validation accuracy reaches **95%** or after **1000 epochs**.

## **Features**

The following 7 features are extracted and normalized to \([0, 1]\):  
**Density** – Proportion of black pixels. (1)  
**Symmetry** – Vertical and diagonal symmetry. (2)  
**Horizontal and Vertical Intersections** – Maximum and average intersections of lines. (3-6)  
**Circular Regions** – Enclosed, connected components (7)  

Features are then input into the Neural Network

## **Input Files**

- **Part 1**:  
  - `train7.csv`, `train9.csv`, `valid7.csv`, `valid9.csv`, `test.csv`  
- **Part 2 & 3**:  
  - `train0.csv`–`train9.csv`, `valid0.csv`–`valid9.csv`, `test1.csv`  


## **How to Run**

1. Place input files in the project directory.  
2. Run the program:  
   ```bash
   python main.py