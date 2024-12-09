# **MNIST Dataset Classifiers: 7 Features**

This repository implements three classifiers for digit classification using the MNIST dataset and **7 extracted features**.

---

## **Overview**

1. **Part 1: Single Perceptron Classifier**  
   A binary perceptron to classify digits **7** and **9** using the pocket algorithm. Outputs optimal weights, error rates, and predictions for the test data.

2. **Part 2: Multi-Class Perceptron**  
   Extends the perceptron to classify all **ten digits (0–9)**. Outputs error rates and predictions for the test data.

3. **Part 3: Neural Network (Backpropagation)**  
   Implements a neural network with:  
   - **One hidden layer** containing **100 neurons**.  
   - **ReLU activation** for the hidden layer.  
   - **Softmax activation** for the output layer to perform multi-class classification.  
   Training stops when validation accuracy reaches **95%** or after **1000 epochs**.

---

## **Features**

The following 7 features are extracted and normalized to \([0, 1]\):  
1. **Density** – Proportion of black pixels.  
2. **Symmetry** – Vertical and diagonal symmetry.  
3. **Horizontal and Vertical Intersections** – Maximum and average intersections of lines.  

---

## **Input Files**

- **Part 1**:  
  - `train7.csv`, `train9.csv`, `valid7.csv`, `valid9.csv`, `test.csv`  
- **Part 2 & 3**:  
  - `train0.csv`–`train9.csv`, `valid0.csv`–`valid9.csv`, `test1.csv`  

---

## **How to Run**

1. Place input files in the project directory.  
2. Run the program:  
   ```bash
   python main.py