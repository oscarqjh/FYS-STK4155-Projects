import numpy as np

# Neural network class
# using batch input
class NeuralNetwork:
    """Neural network class
    
    Args:
        network_input_size: Number of input features
        layer_output_sizes: List of layer sizes
        activation_functions: List of activation functions
        activation_derivatives: List of activation function derivatives
        cost_function: Cost function
        cost_derivative: Cost function derivative
        optimizer: Optimizer
        debug: Debug mode
    """
    def __init__(
            self,
            network_input_size,  # Number of input features
            layer_output_sizes,  # List of layer sizes
            activation_functions,  # List of activation functions
            activation_derivatives,  # List of activation function derivatives
            cost_function,  # Cost function
            cost_derivative,  # Cost function derivative
            optimizer,  # Optimizer
            debug=False  # Debug mode
    ):
        """Initialize the neural network
        
        Args:
            network_input_size: Number of input features
            layer_output_sizes: List of layer sizes
            activation_functions: List of activation functions
            activation_derivatives: List of activation function derivatives
            cost_function: Cost function
            cost_derivative: Cost function derivative
            optimizer: Optimizer
            debug: Debug mode
        """
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_functions = activation_functions
        self.activation_derivatives = activation_derivatives
        self.cost_function = cost_function
        self.cost_derivative = cost_derivative
        self.debug = debug
        self.layers = self._create_layers()

        self.optimizer = optimizer
        self.optimizer.initialize_velocity(self.layers)

    def _create_layers(self):
        """Create the layers of the neural network

        Returns:
            List of tuples (W, b) where W is the weight matrix and b is the bias vector
        """
        layers = []

        i_size = self.network_input_size
        for layer_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_size)
            b = np.random.randn(layer_size)
            layers.append((W, b))

            i_size = layer_size

        return layers
    
    def _forward(self, X):
        """Forward pass of the neural network

        Args:
            X: Input data

        Returns:
            Tuple of layer inputs, zs, and the output of the neural network
        """
        layer_inputs = []
        zs = []
        a = X
        for (W, b), activation_func in zip(self.layers, self.activation_functions):
            layer_inputs.append(a)  
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)  

        return layer_inputs, zs, a
    
    def predict(self, X):
        """Predict the output of the neural network

        Args:
            X: Input data

        Returns:
            Predicted output
        """
        if len(X.shape) == 1:
            a = np.reshape(X, (1, -1))
        else:
            a = X

        _, _, y_pred = self._forward(a)
        return y_pred
    
    def _cost(self, X, targets):
        """Compute the cost of the neural network

        Args:
            X: Input data
            targets: Target values

        Returns:
            Cost of the neural network
        """
        y_pred = self.predict(X)
        return self.cost_function(y_pred, targets)
    
    def _backward(self, X, targets):
        """Backward pass of the neural network

        Args:
            X: Input data
            targets: Target values

        Returns:
            Gradients for each layer
        """
        layer_inputs, zs, y_pred = self._forward(X)
        layer_gradients = [() for layer in self.layers]  # store the gradients for each layer

        # loop over the layers backward
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_deriv = layer_inputs[i], zs[i], self.activation_derivatives[i]
            if i == len(self.layers) - 1:
                # output layer
                delta = self.cost_derivative(y_pred, targets) * activation_deriv(z)

            else:
                # hidden layer
                W, b = self.layers[i+1]
                delta = (delta @ W.T) * activation_deriv(z)

            dW = layer_input.T @ delta
            db = np.mean(delta, axis=0)
            layer_gradients[i] = (dW, db)

        return layer_gradients

    def fit(self, X, y, epochs=1000, return_accuracy=False, **kwargs):
        """Fit the neural network to the data

        Args:
            X: Input data
            y: Target values
            epochs: Number of epochs
            return_accuracy: Return the accuracy score

        Returns:
            List of losses
        """
        if return_accuracy:
            self.return_accuracy = return_accuracy
        return self.optimizer.update(self, X, y, epochs)
    
    def _accuracy(self, y_true, y_pred):
        """Accuracy score used for classification tasks
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return np.sum(y_true == y_pred) / len(y_true)
    
    def num_to_convergence(self, mses, threshold=0.01):
        """Given a list of mse values, return the number of iterations to convergence
            The model converges when the difference between 10 consecutive mse values is less than the threshold
        
            Args:
                mses: list of mean squared errors
                threshold: threshold for the variance of the mean squared error

            Returns:
                number of iterations to convergence
        """
        i = 0
        while True:
            if i >= len(mses)-10:
                return i
            if np.abs(mses[i] - mses[i+10]) < threshold:
                return i
            i += 1

class NNBasicOptimizer:
    """Basic optimizer class

    Args:
        learning_rate: Learning rate
        momentum: Use momentum
        momentum_delta: Momentum delta
    """
    def __init__(self, learning_rate=0.01, momentum=False, **kwargs):
        """Initialize the optimizer

        Args:
            learning_rate: Learning rate
            momentum: Use momentum
            momentum_delta: Momentum delta
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_delta = kwargs.get("momentum_delta", 0.9)
        self.velocity_W = []  # store the velocity for each layer
        self.velocity_b = []  # store the velocity for each layer
    
    def initialize_velocity(self, layers):
        """Initialize the velocity for each layer
        
        Args:
            layers: List of layers
        """
        for W, b in layers:
            self.velocity_W.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))

    def update(self, model, X, y, epochs):
        """Update the weights of the neural network

        Args:
            model: Neural network model
            X: Input data
            y: Target values
            epochs: Number of epochs

        Returns:
            List of losses
        """
        losses = []
        for epoch in range(epochs):
            gradients = model._backward(X, y)
            
            # update the weights
            self._update_weights(model.layers, gradients)

            if model.debug and epoch % 100 == 0:
                loss = model._cost(X, y)
                print(f"Epoch {epoch}: loss = {loss}")
            losses.append(model._cost(X, y))

        return losses

    def _update_weights(self, layers, gradients):
        """Update the weights of the neural network

        Args:
            layers: List of layers
            gradients: Gradients for each layer
        """
        for layer_idx, ((W, b), (dW, db)) in enumerate(zip(layers, gradients)):
            if self.momentum:
                new_velocity_W = self.momentum_delta * self.velocity_W[layer_idx] - self.learning_rate * dW
                new_velocity_b = self.momentum_delta * self.velocity_b[layer_idx] - self.learning_rate * db
                new_W = W + new_velocity_W
                new_b = b + new_velocity_b
                self.velocity_W[layer_idx] = new_velocity_W
                self.velocity_b[layer_idx] = new_velocity_b
            else:
                new_W = W - self.learning_rate * dW
                new_b = b - self.learning_rate * db

            layers[layer_idx] = (new_W, new_b)
    
class NNSGDOptimizer(NNBasicOptimizer):
    """Stochastic gradient descent optimizer class

    Args:
        learning_rate: Learning rate
        momentum: Use momentum
        momentum_delta: Momentum delta
    """
    def __init__(self, learning_rate=0.01, momentum=False, batch_size=32, **kwargs):
        """Initialize the optimizer

        Args:
            learning_rate: Learning rate
            momentum: Use momentum
            momentum_delta: Momentum delta
        """
        super().__init__(learning_rate=learning_rate, momentum=momentum, **kwargs)
        self.batch_size = batch_size
        self.t0 = kwargs.get("t0", 1)
        self.t1 = kwargs.get("t1", 100)

    def _lr_scheduler(self, t):
        """Learning rate scheduler

        Args:
            t: Time step

        Returns:
            Learning rate
        """
        return self.t0 / (t + self.t1)
    
    def update(self, model, X, y, epochs):
        """Update the weights of the neural network

        Args:
            model: Neural network model
            X: Input data
            y: Target values
            epochs: Number of epochs

        Returns:
            List of losses
        """
        n, _ = X.shape
        losses = []
        m = int(n / self.batch_size)

        # loop over the epochs
        for epoch in range(epochs):
            batch_losses = []
            
            # loop over the mini-batches
            for i in range(m):
                random_index = self.batch_size*np.random.randint(m)
                X_mini = X[random_index:random_index+self.batch_size]
                y_mini = y[random_index:random_index+self.batch_size]

                gradients = model._backward(X_mini, y_mini)

                # update the weights
                self._update_weights(model.layers, gradients, epoch=epoch, m=m)

                batch_losses.append(model._cost(X_mini, y_mini))

            if model.debug and epoch % 100 == 0:
                loss = model._cost(X, y)
                print(f"Epoch {epoch}: loss = {loss}")
            losses.append(np.mean(batch_losses))

        return losses
    
    def _update_weights(self, layers, gradients, **kwargs):
        """Update the weights of the neural network

        Args:
            layers: List of layers
            gradients: Gradients for each layer
        """
        epoch = kwargs.get("epoch", 0)
        m = kwargs.get("m", 1)
        self.learning_rate = self._lr_scheduler(epoch*m + i)  # update the learning rate
        return super()._update_weights(layers, gradients)

class NNAdagradOptimizer(NNSGDOptimizer):
    """Adagrad optimizer class

    Args:
        learning_rate: Learning rate
        momentum: Use momentum
        momentum_delta: Momentum delta
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum=False, batch_size=32, **kwargs):
        """Initialize the optimizer

        Args:
            learning_rate: Learning rate
            momentum: Use momentum
            momentum_delta: Momentum delta
        """
        super().__init__(learning_rate=learning_rate, momentum=momentum, batch_size=batch_size, **kwargs)
        self.epsilon = epsilon
        self.giter_W = []  # store the squared gradients for each layer
        self.giter_b = []  # store the squared gradients for each layer

    def initialize_velocity(self, layers):
        """Initialize the velocity for each layer

        Args:
            layers: List of layers
        """
        for W, b in layers:
            self.velocity_W.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))
            self.giter_W.append(np.zeros_like(W))
            self.giter_b.append(np.zeros_like(b))

    def _update_weights(self, layers, gradients, **kwargs):
        """Update the weights of the neural network

        Args:
            layers: List of layers
            gradients: Gradients for each layer
        """
        for layer_idx, ((W, b), (dW, db)) in enumerate(zip(layers, gradients)):
            self.giter_W[layer_idx] += dW**2
            self.giter_b[layer_idx] += db**2

            if self.momentum:
                new_velocity_W = self.momentum_delta * self.velocity_W[layer_idx] + self.learning_rate * dW / (np.sqrt(self.giter_W[layer_idx]) + self.epsilon)
                new_velocity_b = self.momentum_delta * self.velocity_b[layer_idx] + self.learning_rate * db / (np.sqrt(self.giter_b[layer_idx]) + self.epsilon)
                new_W = W - new_velocity_W
                new_b = b - new_velocity_b
                self.velocity_W[layer_idx] = new_velocity_W
                self.velocity_b[layer_idx] = new_velocity_b
            else:
                new_W = W - self.learning_rate * dW / (np.sqrt(self.giter_W[layer_idx]) + self.epsilon)
                new_b = b - self.learning_rate * db / (np.sqrt(self.giter_b[layer_idx]) + self.epsilon)

            layers[layer_idx] = (new_W, new_b)

class NNRMSpropOptimizer(NNSGDOptimizer):
    """RMSprop optimizer class

    Args:
        learning_rate: Learning rate
        rho: Rho
        epsilon: Epsilon
        momentum: Use momentum
        momentum_delta: Momentum delta
    """
    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1e-8, momentum=False, batch_size=32, **kwargs):
        """Initialize the optimizer

        Args:
            learning_rate: Learning rate
            rho: Rho
            epsilon: Epsilon
            momentum: Use momentum
            momentum_delta: Momentum delta
        """
        super().__init__(learning_rate=learning_rate, momentum=momentum, batch_size=batch_size, **kwargs)
        self.rho = rho
        self.epsilon = epsilon
        self.giter_W = []  # store the squared gradients for each layer
        self.giter_b = []  # store the squared gradients for each layer

    def initialize_velocity(self, layers):
        """Initialize the velocity for each layer

        Args:
            layers: List of layers
        """
        for W, b in layers:
            self.velocity_W.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))
            self.giter_W.append(np.zeros_like(W))
            self.giter_b.append(np.zeros_like(b))

    def _update_weights(self, layers, gradients, **kwargs):
        """Update the weights of the neural network

        Args:
            layers: List of layers
            gradients: Gradients for each layer
        """
        for layer_idx, ((W, b), (dW, db)) in enumerate(zip(layers, gradients)):
            self.giter_W[layer_idx] = self.rho * self.giter_W[layer_idx] + (1 - self.rho) * dW**2
            self.giter_b[layer_idx] = self.rho * self.giter_b[layer_idx] + (1 - self.rho) * db**2

            if self.momentum:
                new_velocity_W = self.momentum_delta * self.velocity_W[layer_idx] + self.learning_rate * dW / (np.sqrt(self.giter_W[layer_idx]) + self.epsilon)
                new_velocity_b = self.momentum_delta * self.velocity_b[layer_idx] + self.learning_rate * db / (np.sqrt(self.giter_b[layer_idx]) + self.epsilon)
                new_W = W - new_velocity_W
                new_b = b - new_velocity_b
                self.velocity_W[layer_idx] = new_velocity_W
                self.velocity_b[layer_idx] = new_velocity_b
            else:
                new_W = W - self.learning_rate * dW / (np.sqrt(self.giter_W[layer_idx]) + self.epsilon)
                new_b = b - self.learning_rate * db / (np.sqrt(self.giter_b[layer_idx]) + self.epsilon)

            layers[layer_idx] = (new_W, new_b)

class NNAdamOptimizer(NNSGDOptimizer):
    """Adam optimizer class

    Args:
        learning_rate: Learning rate
        beta1: Beta1
        beta2: Beta2
        epsilon: Epsilon
        momentum: Use momentum
        momentum_delta: Momentum
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=False, batch_size=32, **kwargs):
        """Initialize the optimizer

        Args:
            learning_rate: Learning rate
            beta1: Beta1
            beta2: Beta2
            epsilon: Epsilon
            momentum: Use momentum
            momentum_delta: Momentum
        """
        super().__init__(learning_rate=learning_rate, momentum=momentum, batch_size=batch_size, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.miter_W = []  # store the first moment for each layer
        self.miter_b = []  # store the first moment for each layer
        self.viter_W = []  # store the second moment for each layer
        self.viter_b = []  # store the second moment for each layer

    def initialize_velocity(self, layers):
        """Initialize the velocity for each layer

        Args:
            layers: List of layers
        """
        for W, b in layers:
            self.velocity_W.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))
            self.miter_W.append(np.zeros_like(W))
            self.miter_b.append(np.zeros_like(b))
            self.viter_W.append(np.zeros_like(W))
            self.viter_b.append(np.zeros_like(b))

    def _update_weights(self, layers, gradients, **kwargs):
        """Update the weights of the neural network

        Args:
            layers: List of layers
            gradients: Gradients for each layer
        """
        epoch = kwargs.get("epoch", 0)
        m = kwargs.get("m", 1)
        for layer_idx, ((W, b), (dW, db)) in enumerate(zip(layers, gradients)):
            self.miter_W[layer_idx] = self.beta1 * self.miter_W[layer_idx] + (1 - self.beta1) * dW
            self.miter_b[layer_idx] = self.beta1 * self.miter_b[layer_idx] + (1 - self.beta1) * db
            self.viter_W[layer_idx] = self.beta2 * self.viter_W[layer_idx] + (1 - self.beta2) * dW**2
            self.viter_b[layer_idx] = self.beta2 * self.viter_b[layer_idx] + (1 - self.beta2) * db**2
            miter_W_hat = self.miter_W[layer_idx] / (1 - self.beta1**(epoch*m + i + 1))
            miter_b_hat = self.miter_b[layer_idx] / (1 - self.beta1**(epoch*m + i + 1))
            viter_W_hat = self.viter_W[layer_idx] / (1 - self.beta2**(epoch*m + i + 1))
            viter_b_hat = self.viter_b[layer_idx] / (1 - self.beta2**(epoch*m + i + 1))

            if self.momentum:
                new_velocity_W = self.momentum_delta * self.velocity_W[layer_idx] + self.learning_rate * miter_W_hat / (np.sqrt(viter_W_hat) + self.epsilon)
                new_velocity_b = self.momentum_delta * self.velocity_b[layer_idx] + self.learning_rate * miter_b_hat / (np.sqrt(viter_b_hat) + self.epsilon)
                new_W = W - new_velocity_W
                new_b = b - new_velocity_b
                self.velocity_W[layer_idx] = new_velocity_W
                self.velocity_b[layer_idx] = new_velocity_b
            else:
                new_W = W - self.learning_rate * miter_W_hat / (np.sqrt(viter_W_hat) + self.epsilon)
                new_b = b - self.learning_rate * miter_b_hat / (np.sqrt(viter_b_hat) + self.epsilon)

            layers[layer_idx] = (new_W, new_b)