# Iris Flower Species Classifier

A machine learning project that classifies Iris flowers into three species based on their sepal and petal measurements.

## Overview

This project predicts the species of an Iris flower using two machine learning models:

* Perceptron
* Multilayer Perceptron (Neural Network)

The three flower species are:

* Setosa
* Versicolor
* Virginica

## Dataset

The project uses the **Iris.csv** dataset containing the following features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
* Species (Target)

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

## Project Workflow

1. Load the dataset
2. Explore and visualize the data
3. Preprocess the data
4. Split the data into training and testing sets
5. Scale the features
6. Train the Perceptron model
7. Train the Neural Network model
8. Evaluate the models

## Results

| Model                | Accuracy |
| -------------------- | -------: |
| Perceptron           |      80% |
| Neural Network (MLP) |      98% |

The Neural Network achieved higher accuracy than the Perceptron model.

## How to Run

### Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Run the project

Open the **Iris_Classification.ipynb** notebook in Jupyter Notebook or Google Colab and run all the cells.

## Project Structure

```
├── Iris.csv
├── Iris_Classification.ipynb
└── README.md
```


