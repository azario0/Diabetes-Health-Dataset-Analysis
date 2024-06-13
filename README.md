# Neural Network Model for Diagnosing Diabetes

This repository contains a Jupyter Notebook for building, training, and evaluating a neural network model to diagnose diabetes based on various features. The notebook uses PyTorch for model implementation and Scikit-learn for data preprocessing and evaluation.

## Prerequisites

Make sure you have the following libraries installed:

- `pandas`
- `torch`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `ipywidgets`
- `ipython`

You can install these libraries using pip:

```bash
pip install pandas torch scikit-learn seaborn matplotlib ipywidgets ipython
```
## Notebook Overview

The notebook follows these steps:

    Data Preparation:
        Importing necessary libraries.
        Dropping irrelevant columns.
        Identifying categorical and numerical features.
        Preprocessing the data by one-hot encoding categorical features and scaling numerical features.

    Data Splitting:
        Splitting the dataset into training and testing sets.

    Converting Data to PyTorch Tensors:
        Converting the preprocessed data to PyTorch tensors.

    Defining the Neural Network Model:
        Building a simple neural network with two hidden layers using PyTorch.

    Training the Model:
        Training the neural network using the training data.

    Evaluating the Model:
        Evaluating the model's performance on both training and testing sets.
        Calculating accuracy and plotting a confusion matrix.

    Interactive Widgets:
        Using interactive widgets to select and display column groups for correlation heatmaps.

Code Explanation
Libraries Imported

    pandas: For data manipulation and analysis.
    torch, torch.nn, torch.optim: For building and training the neural network.
    scikit-learn:
        train_test_split: For splitting the dataset into training and testing sets.
        StandardScaler, OneHotEncoder: For scaling numerical features and one-hot encoding categorical features.
        ColumnTransformer: For applying different preprocessing steps to different feature columns.
    seaborn, matplotlib.pyplot: For data visualization.
    ipywidgets, IPython.display: For creating and displaying interactive widgets in Jupyter Notebooks.
    accuracy_score, confusion_matrix: For evaluating the model's performance.

Usage

    Load the Data:
        Ensure the dataset is loaded into a DataFrame df.

    Run the Cells:
        Follow the sequential cells in the notebook to preprocess the data, build the model, train it, and evaluate it.

    Visualize Results:
        Use the interactive dropdown to visualize correlation heatmaps for different column groups.
        View the confusion matrix to understand the model's performance.

Conclusion

This notebook demonstrates a complete workflow for building a neural network model to diagnose diabetes. It covers data preprocessing, model training, evaluation, and visualization, providing a comprehensive guide for similar machine learning tasks.