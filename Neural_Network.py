import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn.functional as F

# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
# ???? nadie tiene una mejor respuesta sin embargo python insiste en molestar
# con este error.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# https://stackoverflow.com/questions/72580800/number-of-digits-in-mnist-data-in-python
class Neural_Network:

    def __init__(self):
        # load train samples
        train_data = np.loadtxt("MNIST_excel/mnist_train.csv", delimiter=",")
        X_train = train_data[:, 1:]
        y_train = train_data[:, 0]

        # load test samples
        test_data = np.loadtxt("MNIST_excel/mnist_test.csv", delimiter=",")
        X_test = test_data[:, 1:]
        t_test = test_data[:, 0]

        unique, counts = np.unique(y_train, return_counts=True)
        dict(zip(unique, counts))
        {0.0: 5923, 1.0: 6742, 2.0: 5958, 3.0: 6131, 4.0: 5842, 5.0: 5421, 6.0: 5918, 7.0: 6265, 8.0: 5851, 9.0: 5949}

        # Reshape digits into 28x28 for Matplotlib to properly print them
        X_train = X_train.reshape(60000, 28, 28)

        # Create a 2D grid of subfigures
        fig, axes = plt.subplots(10, 9, figsize=(15, 18))
        numbers = np.array([9, 10000])
        # i is the digit iterator. j is the samples-per-digit iterator
        for i in range(10):
            # Select digits that equal "i"
            digits = X_train[y_train == i]
            for j in range(9):
                # Select axis, and print the digit on it!
                ax = axes[i, j]
                numbers[i, ax]
                ax.imshow(digits[j], cmap="gray")
            plt.text(f"number {i}: {numbers[ax]}")
        # This will give you the visualization
        plt.show()
