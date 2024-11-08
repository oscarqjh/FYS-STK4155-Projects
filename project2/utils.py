import numpy as np

def f(x):
    """Function to generate the data

        Args:
            x: input data
        
        Returns:
            y: output data (4 + 3*x + 5*x^2 + noise)
    """
    return 4 + 3*x + 5*x**2 + np.random.randn(len(x),1)  # shape (n,1) matrix, n random numbers from a normal distribution

def generate_data(n):
    """Function to generate the data

        Args:
            n: number of data points
        
        Returns:
            x: input data
            y: output data
    """
    x = 2*np.random.rand(n,1)  # shape (n,1) matrix, n random numbers between 0 and 2
    return x, f(x)

# define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_ReLU_der(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def mse(predict, target):
    return np.mean((predict - target)**2)

def mse_derivative(predict, target):
    return 2 * (predict - target) / len(predict)

# define some helper functions
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def cross_entropy(predict, target):
    epsilon = 1e-12
    predict = np.clip(predict, epsilon, 1. - epsilon)
    return -np.sum(target * np.log(predict)) / target.shape[0]

def cross_entropy_derivative(predict, target):
    # applicable for softmax + cross entropy
    return predict - target

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

# See https://stackoverflow.com/questions/58461808/understanding-backpropagation-with-softmax
# from the above discussion, seems like its common practise to pass the derivative of the loss function instead
# hence, softmax_der should not modify the derivative
def softmax_der(z):
    da_dz = np.ones(z.shape)
    return da_dz

# Define cost function for binary classification logistic regression with l2 regularization
def logistic_cost_function(predict, target, weights, l2_lambda=0.01):
    epsilon = 1e-12
    predict = np.clip(predict, epsilon, 1. - epsilon)
    return -np.sum(target * np.log(predict)) / target.shape[0] + 0.5 * l2_lambda * np.sum(weights**2)

# Define derivative of cost function for binary classification logistic regression with l2 regularization
def logistic_cost_derivative(predict, target, weights, l2_lambda=0.01):
    return predict - target + l2_lambda * weights