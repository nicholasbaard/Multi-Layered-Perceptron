import argparse
from mlp_global import MLPFstar
from process_data import get_mnist
from multi_layer_perceptron import MLPBP
from utils import *

# Parse command line arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=list[int], default=[128,64], help='List of hidden layer dimensions')
    parser.add_argument('--dim', type=list[int], default=[784,128,64,10], help='List of hidden layer dimensions')
    parser.add_argument('--activation', type=str, default="sigmoid", help='Which activation function to use. Choices: sigmoid, tanh, relu')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Epsilon value')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lambda_reg', type=float, default=1e-2, help='Lambda (L2) regularization value')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')

    args = parser.parse_args()
    
    # Load MNIST dataset
    
    X_train_Fstar, X_test_Fstar, y_train_Fstar, y_test_Fstar = get_mnist()
    X_train_BP, X_test_BP, y_train_BP, y_test_BP = X_train_Fstar.T, X_test_Fstar.T, y_train_Fstar.T, y_test_Fstar.T
    # Initialize MLP modelwith Backpropagation
    mlp_bp = MLPBP(input_dim=X_train_BP.shape[1], hidden_dim=args.hidden_dim, output_dim=y_train_BP.shape[1])
    # Initialize MLP model with F-adjoint
    mlp_fstar = MLPFstar(dim=args.dim)
   # Train the MLP_BP model
    train_loss_mlp_BP, test_loss_mlp_BP, train_accuracy_mlp_BP, test_accuracy_mlp_BP = mlp_bp.train(X_train=X_train_BP, 
                                                                     y_train=y_train_BP, 
                                                                     X_test=X_test_BP, 
                                                                     y_test=y_test_BP, 
                                                                     epochs=args.epochs, 
                                                                     alpha=args.alpha,
                                                                     epsilon=args.epsilon, 
                                                                     batch_size=args.batch_size,
                                                                     lambda_reg=args.lambda_reg
                                                                     )

    # Train the MLP_Fstar model
    train_loss_mlp_Fstar, test_loss_mlp_Fstar, train_accuracy_mlp_Fstar, test_accuracy_mlp_Fstar = mlp_fstar.train(X_train=X_train_Fstar, 
                                                                     y_train=y_train_Fstar, 
                                                                     y_test=y_test_Fstar,
                                                                     X_test=X_test_Fstar,  
                                                                     epochs=args.epochs, 
                                                                     alpha=args.alpha,
                                                                     epsilon=args.epsilon, 
                                                                     batch_size=args.batch_size,
                                                                     lambda_reg=args.lambda_reg
                                                                     )
    
     # Plot the results for mlp_BP
    plot_results_bp(mlp_bp, X_test_BP, y_test_BP, train_loss_mlp_BP, test_loss_mlp_BP, train_accuracy_mlp_BP, test_accuracy_mlp_BP, show_plots=args.show_plots)
    # Plot the results for mlp_Fstar
    plot_results_fstar(mlp_fstar, X_test_Fstar, y_test_Fstar, train_loss_mlp_Fstar, test_loss_mlp_Fstar, train_accuracy_mlp_Fstar, test_accuracy_mlp_Fstar, show_plots=args.show_plots)

if __name__ == "__main__":
    main()
