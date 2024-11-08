import numpy as np
import torch

class TorchLinearModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(TorchLinearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x): 
        return self.linear(x)

class TorchLinearRegression:
    def __init__(
            self, 
            model,  # torch model
            loss_fn,  # torch loss function
            optimizer = None,  # torch optimizer
            degree=2
        ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.degree = degree

    def fit(self, x, y, epochs=1000):
        x_torch = torch.tensor(x, dtype=torch.float32)
        X = self._create_polynomial_features(x_torch)
        y = torch.tensor(y, dtype=torch.float32)

        if self.optimizer is None:
            self._fit_default(X, y, epochs)
        else:
            self.mses = []
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()

                mse = loss.item()
                self.mses.append(mse)

        return self
    
    def _fit_default(self, X, y, epochs):
        self.mses = []
        for epoch in range(epochs):
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)

            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

            mse = loss.item()
            self.mses.append(mse)

    def _create_polynomial_features(self, x):
        """Generate polynomial features up to a specified degree."""
        return torch.cat([x ** i for i in range(1, self.degree + 1)], dim=1)


# Define the feed-forward neural network with layers 16, 16, 1
class SimpleFFNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleFFNN, self).__init__()
        # Define the layers
        self.fc1 = torch.nn.Linear(input_size, 16)  # First hidden layer with 16 neurons
        self.sigmoid1 = torch.nn.Sigmoid()  # Sigmoid activation function for the first hidden layer
        self.fc2 = torch.nn.Linear(16, 16)  # Second hidden layer with 16 neurons
        self.sigmoid2 = torch.nn.Sigmoid()  # Sigmoid activation function for the second hidden layer
        self.fco = torch.nn.Linear(16, 1)  # Output layer with 1 neuron
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation for output layer

    def forward(self, x):
        out = self.fc1(x)  # First hidden layer
        out = self.sigmoid1(out)  # Apply sigmoid activation
        out = self.fc2(out)  # Second hidden layer
        out = self.sigmoid2(out)  # Apply sigmoid activation
        out = self.fco(out)  # Output layer
        out = self.leaky_relu(out)  # Apply leaky ReLU activation to output
        return out
    
class TorchNeuralNetwork:
    def __init__(
            self,
            model,
            loss_fn,
            optimizer=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(self, X, y, epochs=100):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        if self.optimizer is None:
            self._fit_default(X, y, epochs)
        else:
            self.mses = []
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()
                self.mses.append(loss.item())

    def _fit_default(self, X, y, epochs):
        self.mses = []
        for epoch in range(epochs):
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass (compute gradients)
            loss.backward()
            
            # Manually update weights using gradient descent
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad  # gradient descent update step
            
            # Zero the gradients after updating
            model.zero_grad()


# Define the logistic regression model
class SimpleLogistic(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleLogistic, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)  # One output unit for binary classification

    def forward(self, x):
        # Apply linear transformation and sigmoid activation for logistic regression
        return torch.sigmoid(self.linear(x))
    
class TorchLogisticRegression:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X, y, epochs=100):
        # Convert data to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = y.reshape(-1, 1)  # Reshape y to match the shape of the output from the model
        y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()