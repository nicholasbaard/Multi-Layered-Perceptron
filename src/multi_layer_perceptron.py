import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from activations import Sigmoid, Softmax, Tanh, Relu

class MLP:
    def __init__(self, input_dim:int, hidden_dim:list[int], output_dim:int, activation:str="sigmoid"):
        # Initialize the input dimension, number of hidden layers, hidden layer dimension, output dimension, learning rate, and activation function
        self.input_dim = input_dim
        self.num_hidden = len(hidden_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        elif activation == "relu":
            self.activation = Relu()

        self.softmax = Softmax()

        # Initialize the weights and biases
        self.__initialize_weights()

    def __initialize_weights(self):
        # Initialize the weights and biases for each layer
        for i in range(1, self.num_hidden + 2):
            # Initialize the weights for the current layer
            w = np.random.randn(self.hidden_dim[i-2] if i > 1 else self.input_dim , self.hidden_dim[i-1] if i < self.num_hidden + 1 else self.output_dim ) 
            w = w * np.sqrt(1./w.shape[0])
            setattr(self, f'W{i}', w)
            
            # Initialize the biases for the current layer
            b = np.zeros((w.shape[1], 1)).reshape(1, -1)
            setattr(self, f'b{i}', b)

    def forward(self, x:np.ndarray) -> np.ndarray:
        # Perform the forward pass through the network
        for i in range(1, self.num_hidden+3):
            if i == 1:
                # Set the input as the initial activation
                a = x
                setattr(self, f'A{i}', a)
            elif i < self.num_hidden + 2:
                # Compute the activations for the hidden layers
                z = np.dot(getattr(self, f'A{i-1}'), getattr(self, f'W{i-1}')) + getattr(self, f'b{i-1}')
                a = self.activation(z)
                setattr(self, f'A{i}', a)
            else:
                # Compute the activations for the output layer using softmax
                z = np.dot(getattr(self, f'A{i-1}'), getattr(self, f'W{i-1}'))
                a = self.softmax(z)
                setattr(self, f'A{i}', a)

        # Return the output activations
        return getattr(self, f'A{self.num_hidden + 2}')

    def backpropagate(self, y:np.ndarray):
        # Calculate the number of training examples
        m = y.shape[0]

        # Initialize gradients
        for i in range(1, self.num_hidden + 2):
            grad = np.zeros(getattr(self, f'W{i}').shape)
            setattr(self, f'grad{i}', grad)

        # Backpropagate and get deltas
        for i in range(self.num_hidden + 2, 1, -1):
            # Calculate delta for the output layer
            if i == self.num_hidden + 2:
                delta = (getattr(self, f'A{i}') - y)
                setattr(self, f'delta{i}', delta)
            # Calculate delta for the hidden layers
            else:
                delta = (np.dot(getattr(self, f'delta{i+1}'),np.transpose(getattr(self, f'W{i}')) )) * self.activation(getattr(self, f'A{i}'), derivative=True)
                setattr(self, f'delta{i}', delta)
    
        # Calculate gradients
        for i in range(self.num_hidden + 1, 1, -1):
            grad = getattr(self, f'grad{i}')
            grad += (1/m)*np.dot(getattr(self, f'A{i}').T, getattr(self, f'delta{i+1}')) + (self.lambda_reg/m * getattr(self, f'W{i}'))
            setattr(self, f'grad{i}', grad)

        # Update weights
        for i in range(1, self.num_hidden + 2):
            weight = getattr(self, f'W{i}')
            weight -= self.alpha * getattr(self, f'grad{i}')
            setattr(self, f'W{i}', weight)
    
    def accuracy(self, y:np.ndarray, y_hat:np.ndarray) -> float:
        # Calculate the accuracy of the model's predictions
        return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))
        
    def cross_entropy_loss(self, y:np.ndarray, output:np.ndarray) -> float:
        # Calculate the cross-entropy loss
        l_sum = np.sum(np.multiply(y, np.log(output)))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l

    def train(self, 
              X_train:np.ndarray, 
              y_train:np.ndarray, 
              X_test:np.ndarray, 
              y_test:np.ndarray, 
              epochs:int=100, 
              alpha:float=0.1, 
              epsilon:float=1e-3, 
              batch_size:int=64, 
              lambda_reg:float=1e-4
              ) -> Tuple[List[float], List[float], List[float], List[float]]:

        self.epochs = epochs
        self.alpha = alpha
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        # Initialize lists to store training and test metrics
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        prev_loss = float('inf')
        num_batches = int(X_train.shape[0]/self.batch_size)
        
        for _ in tqdm(range(self.epochs), desc="Training"):
            # Shuffle the training data
            perm = np.random.permutation(X_train.shape[0])
            x_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for i in range(num_batches):
                # Get a batch of training data
                x_batch = x_shuffled[i*self.batch_size:min((i+1)*self.batch_size, X_train.shape[0]-1)]
                y_batch = y_shuffled[i*self.batch_size:min((i+1)*self.batch_size, X_train.shape[0]-1)]

                # Forward pass and backpropagation
                outputs = self.forward(x_batch)
                self.backpropagate(y_batch)

            # Compute training metrics
            outputs = self.forward(X_train)
            curr_loss = self.cross_entropy_loss(y=y_train, output=outputs)
            train_loss.append(curr_loss)
            train_accuracy.append(self.accuracy(y_train, outputs))

            # Compute test metrics
            test_outputs = self.forward(X_test)
            test_l = self.cross_entropy_loss(y=y_test, output=test_outputs)
            test_loss.append(test_l)
            test_accuracy.append(self.accuracy(y_test, test_outputs))

            # Early stopping condition
            if abs(curr_loss - prev_loss) < self.epsilon:
                break
            prev_loss = curr_loss
            
        return train_loss, test_loss, train_accuracy, test_accuracy
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        outputs = self.forward(x)
        return np.argmax(outputs, axis=1)
    