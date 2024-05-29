import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from activations import Sigmoid, Softmax, Tanh, Relu

class MLP:
    def __init__(self,  dim:list[int] , activation:str="sigmoid"):
        # Initialize the input dimension, number of hidden layers, hidden layer dimension, output dimension, learning rate, and activation function
        self.dim = dim
        self.num_hidden = len(dim)
      

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
        for i in range(1, self.num_hidden ):
            # Initialize the weights for the current layer
            w = np.random.randn(self.dim[i] , self.dim[i-1]+1  ) 
            w = w * np.sqrt(1./w.shape[1])
            setattr(self, f'W{i}', w)            

    def F_pass(self, x:np.ndarray) -> np.ndarray:
        # Perform the forward pass through the network
        for i in range(1, self.num_hidden):
            
            # Set the input as the initial activation
            X0 = np.vstack((x, np.ones((1,x.shape[1]))))
            setattr(self, f'X{0}', X0)
            if i!= self.num_hidden-1:
                # Compute the activations for the hidden layers
                Yi = np.dot(getattr(self, f'W{i}'), getattr(self, f'X{i-1}')) 
                setattr(self, f'Y{i}', Yi)
                Xi = np.vstack((self.activation(Yi), np.ones((1,self.activation(Yi).shape[1]))))
                setattr(self, f'X{i}', Xi)
            else:
                # Compute the activations for the output layer using softmax
                YL = np.dot(getattr(self, f'W{i}'), getattr(self, f'X{i-1}')) 
                setattr(self, f'Y{i}', YL)
                XL = self.softmax(YL)
                setattr(self, f'X{i}', XL)

        # Return the output activations
        return  getattr(self, f'X{self.num_hidden-1}') 
    
    def F_star_pass(self, y:np.ndarray):
        # Calculate the number of training examples
        m = y.shape[1]

        # Initialize gradients
        for i in range(1, self.num_hidden):
            grad = np.zeros(getattr(self, f'W{i}').shape)
            setattr(self, f'grad{i}', grad)

        # Backpropagate and get deltas
        for i in range(self.num_hidden , 2, -1):
            # Calculate delta for the output layer
     
            xstar = (getattr(self, f'X{self.num_hidden-1}') - y)
            
            ystar=xstar* self.activation(getattr(self, f'Y{self.num_hidden-1}'), derivative=True)
            setattr(self, f'Ystar{self.num_hidden-1}', ystar)
            # Calculate delta for the hidden layers
        
            xstar = np.dot(np.transpose(getattr(self, f'W{i-1}')[:,:-1]), getattr(self, f'Ystar{i-1}') )
            ystar=xstar* self.activation(getattr(self, f'Y{i-2}'), derivative=True)
            setattr(self, f'Ystar{i-2}', ystar)
        self.lambda_reg=1e-3
        # Calculate gradients
        for i in range(self.num_hidden-1, 0, -1):
            grad = getattr(self, f'grad{i}')
            grad += (1/m)*np.dot(getattr(self, f'Ystar{i}'), getattr(self, f'X{i-1}').T) + (self.lambda_reg/m * getattr(self, f'W{i}'))
            setattr(self, f'grad{i}', grad)
        self.alpha=0.1
        # Update weights
        for i in range(1, self.num_hidden):
            weight = getattr(self, f'W{i}')
            weight -= self.alpha * getattr(self, f'grad{i}')
            setattr(self, f'W{i}', weight)

    def accuracy(self, y:np.ndarray, y_hat:np.ndarray) -> float:
        # Calculate the accuracy of the model's predictions
        return np.mean(np.argmax(y, axis=0) == np.argmax(y_hat, axis=0))
        
    def cross_entropy_loss(self, y:np.ndarray, output:np.ndarray) -> float:
        # Calculate the cross-entropy loss
        l_sum = np.sum(np.multiply(y, np.log(output)))
        m = y.shape[1]
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
        num_batches = int(X_train.shape[1]/self.batch_size)
        
        for _ in tqdm(range(self.epochs), desc="Training"):
            # Shuffle the training data
            perm = np.random.permutation(X_train.shape[1])
            x_shuffled = X_train[:,perm]
            y_shuffled = y_train[:,perm]

            for i in range(num_batches):
                # Get a batch of training data
                x_batch = x_shuffled[:, i*self.batch_size:min((i+1)*self.batch_size, X_train.shape[1]-1)]
                y_batch = y_shuffled[:, i*self.batch_size:min((i+1)*self.batch_size, X_train.shape[1]-1)]

                # F and Fstar passes
                outputs = self.F_pass(x_batch)
                self.F_star_pass(y_batch)

            # Compute training metrics
            outputs = self.F_pass(X_train)
            curr_loss = self.cross_entropy_loss(y=y_train, output=outputs)
            train_loss.append(curr_loss)
            train_accuracy.append(self.accuracy(y_train, outputs))

            # Compute test metrics
            test_outputs = self.F_pass(X_test)
            test_l = self.cross_entropy_loss(y=y_test, output=test_outputs)
            test_loss.append(test_l)
            test_accuracy.append(self.accuracy(y_test, test_outputs))

            # Early stopping condition
            if abs(curr_loss - prev_loss) < self.epsilon:
                break
            prev_loss = curr_loss
            
        return train_loss, test_loss, train_accuracy, test_accuracy
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        outputs = self.F_pass(x)
        return np.argmax(outputs, axis=0)