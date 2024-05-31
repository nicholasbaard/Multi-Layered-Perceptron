from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import numpy as np
from typing import Tuple

from utils import one_hot_encode


def get_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Download the MNIST dataset
    # Ensure the directory exists
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)

    mnist = fetch_openml('mnist_784', 
                         version=1, 
                         as_frame=False, 
                         return_X_y=True, 
                         data_home=data_dir
                         )

    # Extract features and labels
    x, y = mnist[0], mnist[1]

    # Reshape the data 
    x = x.reshape(-1, 28*28)

    # Normalize the data to the range [0, 1]
    x = x / 255.0
   

    # Convert labels to integers and one hot encode
    y = y.astype(int).reshape(-1, 1)
    y = one_hot_encode(y)
   

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=19)
    x_train, x_test, y_train, y_test =x_train.T, x_test.T, y_train.T, y_test.T       
    # Verify the shape of the data
    print("\nData shape:")
    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}\n")

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    
    get_mnist()