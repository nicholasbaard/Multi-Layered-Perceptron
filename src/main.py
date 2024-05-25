import argparse

from process_data import get_mnist
from multi_layer_perceptron import MLP
from utils import plot_results

# Parse command line arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=list[int], default=[128,64], help='List of hidden layer dimensions')
    parser.add_argument('--activation', type=str, default="sigmoid", help='Which activation function to use. Choices: sigmoid, tanh, relu')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Epsilon value')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lambda_reg', type=float, default=1e-3, help='Lambda (L2) regularization value')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')

    args = parser.parse_args()
    
    # Load MNIST dataset
    X_train, X_test, y_train, y_test = get_mnist()

    # Initialize MLP model
    mlp = MLP(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim, output_dim=y_train.shape[1])

    # Train the MLP model
    train_loss, test_loss, train_accuracy, test_accuracy = mlp.train(X_train=X_train, 
                                                                     y_train=y_train, 
                                                                     X_test=X_test, 
                                                                     y_test=y_test, 
                                                                     epochs=args.epochs, 
                                                                     alpha=args.alpha,
                                                                     epsilon=args.epsilon, 
                                                                     batch_size=args.batch_size,
                                                                     lambda_reg=args.lambda_reg
                                                                     )
    
    # Plot the results
    plot_results(mlp, X_test, y_test, train_loss, test_loss, train_accuracy, test_accuracy, show_plots=args.show_plots)

if __name__ == "__main__":
    main()
