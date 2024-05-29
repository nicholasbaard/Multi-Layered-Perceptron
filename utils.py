import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def one_hot_encode(y:np.ndarray, num_classes=10) -> np.ndarray:

    # Flatten the array to get the shape (n,)
    y = y.flatten()

    # Number of classes
    num_classes = 10

    # Initialize the one-hot encoded array
    y_one_hot = np.zeros((y.size, num_classes))

    # Set the appropriate elements to 1
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


def plot_results(mlp, 
                 X_test:np.ndarray, 
                 y_test:np.ndarray, 
                 train_loss:list[float], 
                 test_loss:list[float], 
                 train_accuracy:list[float], 
                 test_accuracy:list[float], 
                 show_plots=True
                 ):
        # Predict the labels for the test data
        y_hat = mlp.predict(X_test)

        # Print the classification report
        print(classification_report(np.argmax(y_test, axis=0), y_hat))

        # Create a figure with 3 subplots
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

        # Plot the train and test loss over time
        ax[0].plot(train_loss, label='Train Loss')
        ax[0].plot(test_loss, label='Test Loss')
        ax[0].set_title('Loss Over Time')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Plot the train and test accuracy over time
        ax[1].plot(train_accuracy, label='Train Accuracy')
        ax[1].plot(test_accuracy, label='Test Accuracy')
        ax[1].set_title('Accuracy Over Time')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()


        # Compute the confusion matrix
        cm = confusion_matrix(np.argmax(y_test, axis=0), y_hat)

        # Plot the confusion matrix heatmap on the second subplot
        sns.heatmap(cm, annot=True, ax=ax[2], cmap='Blues')
        ax[2].set_title('Test Set Confusion Matrix')
        ax[2].set_xlabel('Predicted Labels')
        ax[2].set_ylabel('True Labels')

        # Display the plots
        plt.tight_layout()
        plt.savefig('mnist_28_05.png')
        if show_plots:
            plt.show()