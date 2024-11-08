import numpy as np

class LinearRegression():
    """Linear regression model
    
        Attributes:
            reg_method: regression method (ols or ridge)
            optimizer: optimizer object
            degree: degree of the polynomial
            ridge_lambda: lambda parameter for ridge regression
            theta: weights of the model
            grads: gradients of the model
            mses: mean squared errors of the model
    """
    def __init__(
            self, 
            reg_method="ols",  # regression method (ols or ridge)
            optimizer=None,  # optimizer object
            degree=2,  # degree of the polynomial
            **kwargs
    ):
        """Initialize the model
        
            Args:
                reg_method: regression method
                optimizer: optimizer object
                degree: degree of the polynomial
                **kwargs: additional parameters
        """
        self.reg_method = reg_method
        self.optimizer = optimizer
        self.degree = degree

        if self.reg_method == "ridge":
            assert "ridge_lambda" in kwargs, "Ridge regression requires a ridge_lambda parameter"
            self.ridge_lambda = kwargs["ridge_lambda"]
        

    def fit(self, x, y, epochs=1000):
        """Fit the model to the data
        
            Args:
                x: input data (n x 1) vector
                y: output data (n x 1) vector
                epochs: number of epochs
        """
        n, _ = x.shape  # len of x
        X = self._design_matrix(x)  # design matrix

        if self.optimizer is not None:
            grads, mses = self.optimizer.fit(self, X, y)
            self.grads = grads
            self.mses = mses
        else:
            self._standard_fit(x, y)
    
    def _standard_fit(self, x, y):
        """Fit the model to the data
        
            Args:
                x: input data (n x 1) vector
                y: output data (n x 1) vector

            Returns:
                self
        """
        n, _ = x.shape  # len of x
        X = self._design_matrix(x)  # design matrix
        if self.reg_method == "ols":
            self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.reg_method == "ridge":
            self.theta = np.linalg.inv(X.T @ X + 2*self.ridge_lambda*np.eye(X.shape[1])) @ X.T @ y
        return self

    def _design_matrix(self, x):
        """Create the design matrix
        
            Args:
                x: input data (n x 1) vector
                degree: degree of the polynomial

            Returns:
                X: design matrix (n x degree+1) matrix
        """
        n, _ = x.shape  # len of x
        X = np.ones((n, 1))
        for i in range(1, self.degree+1):
            X = np.c_[X, x**i]
        return X

    def predict(self, x):
        """Predict the output
        
            Args:
                x: input data (n x 1) vector

            Returns:
                y: output data (n x 1) vector
        """
        X = self._design_matrix(x)
        return X @ self.theta
    
    def _mse(self, x, y):
        """Calculate the mean squared error
        
            Args:
                x: input data (n x 1) vector
                y: output data (n x 1) vector

            Returns:
                mse: mean squared error
        """
        return np.mean((self.predict(x) - y)**2)   
    
    def analyse(self, x, y, threshold=0.0025, max_iterations=1000):
        """Analyse the model with the mean squared error
        
            Args:
                x: input data (n x 1) vector
                y: output data (n x 1) vector
                threshold: threshold for the variance of the mean squared error
                max_iterations: maximum number of iterations

            Returns:
                mean of the mean squared errors
        """
        mses = []
        i = 0
        while True:
            mses.append(self._mse(x, y))

            if i > max_iterations:
                print("Max iterations reached")
                break
            if len(mses) > 100:
                mse_var = np.var(mses)
                if mse_var < threshold:
                    break

        return np.mean(mses)
    
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
    
#####################################################################
############################ Optimizers #############################
#####################################################################


class GradientDescentOptimizer():
    """Gradient descent optimizer
    
        Attributes:
            learning_rate: learning rate
            epochs: number of epochs
            momentum: momentum
            momentum_delta: momentum delta
    """
    def __init__(
            self,
            learning_rate=0.1,  # learning rate
            epochs=1000,  # number of epochs
            momentum=False,  # momentum
            **kwargs
    ):
        """Initialize the optimizer
        
            Args:
                learning_rate: learning rate
                epochs: number of epochs
                momentum: momentum
                **kwargs: additional parameters
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum

        if self.momentum:
            assert "momentum_delta" in kwargs, "Momentum requires a momentum_delta parameter"
            self.momentum_delta = kwargs["momentum_delta"] 

    def fit(self, model, X, y):
        """Fit the model to the data
        
            Args:
                model: model object
                X: design matrix (n x degree+1) matrix
                y: output data (n x 1) vector

            Returns:
                gradients_history: history of the gradients
                mse_history: history of the mean squared errors
        """
        n, _ = X.shape  # len of x
        model.theta = np.random.randn(X.shape[1],1)  # random initialization of the parameters
        gradients_history = np.zeros((self.epochs, X.shape[1]))  # history of the gradients
        mse_history = np.zeros(self.epochs)  # history of the mean squared errors
        change = 0.0
        for i in range(self.epochs):
            regularization = 0 if model.reg_method == "ols" else 2*model.ridge_lambda*model.theta
            gradients = 2/n * X.T @ (X @ model.theta - y) + regularization

            if self.momentum:
                new_change = self.learning_rate * gradients + self.momentum_delta * change
                model.theta -= new_change
                change = new_change
            else:
                model.theta -= self.learning_rate * gradients

            gradients_history[i] = gradients.flatten()
            mse_history[i] = self._mse(X, y, model)

        return gradients_history, mse_history
    
    def _mse(self, X, y, model):
        """Calculate the mean squared error
        
            Args:
                X: design matrix (n x degree+1) matrix
                y: output data (n x 1) vector

            Returns:
                mse: mean squared error
        """
        return np.mean((X @ model.theta - y)**2)
    
class SGDOptimizer(GradientDescentOptimizer):
    """Stochastic gradient descent optimizer
    
        Attributes:
            batch_size: batch size
    """
    def __init__(
            self, 
            learning_rate=0.1, 
            epochs=1000, 
            momentum=False, 
            batch_size=32,
            **kwargs
        ):
        """Initialize the optimizer
        
            Args:
                learning_rate: learning rate
                epochs: number of epochs
                momentum: momentum
                batch_size: batch size
                **kwargs: additional parameters
        """
        super().__init__(learning_rate, epochs, momentum, **kwargs)
        self.batch_size = batch_size
        self.t0 = kwargs.get("t0", 1)
        self.t1 = kwargs.get("t1", 1000)

    def _learning_schedule(self, t):
        """Learning schedule
        
            Args:
                t: iteration number

            Returns:
                learning rate
        """
        t0, t1 = self.t0, self.t1
        return t0 / (t + t1)
    
    def fit(self, model, X, y):
        """Fit the model to the data

            Args:
                model: model object
                X: design matrix (n x degree+1) matrix
                y: output data (n x 1) vector

            Returns:
                gradients_history: history of the gradients
                mse_history: history of the mean squared errors
        """
        n, _ = X.shape  # len of x
        model.theta = np.random.randn(X.shape[1],1)  # random initialization of the parameters
        gradients_history = np.zeros((self.epochs, X.shape[1]))  # history of the gradients
        mse_history = np.zeros(self.epochs)  # history of the mean squared errors
        change = 0.0
        m = int(n / self.batch_size)  # number of mini-batches
        for epoch in range(self.epochs):
            batch_gradient = np.zeros((m, X.shape[1]))
            for i in range(m):
                random_index = self.batch_size*np.random.randint(m)
                X_mini = X[random_index:random_index+self.batch_size]
                y_mini = y[random_index:random_index+self.batch_size]

                regularization = 0 if model.reg_method == "ols" else 2*model.ridge_lambda*model.theta
                gradients = 2/self.batch_size * X_mini.T @ (X_mini @ model.theta - y_mini) + regularization

                self.learning_rate = self._learning_schedule(epoch*m + i)

                if self.momentum:
                    new_change = self.learning_rate * gradients + self.momentum_delta * change
                    model.theta -= new_change
                    change = new_change
                else:
                    model.theta -= self.learning_rate * gradients

                batch_gradient[i] = gradients.flatten()

            gradients_history[epoch] = np.mean(batch_gradient, axis=0)
            mse_history[epoch] = self._mse(X, y, model)

        return gradients_history, mse_history
    
class AdagradOptimizer(SGDOptimizer):
    """Adagrad optimizer
    
        Attributes:
            epsilon: epsilon 
    """
    def __init__(self, learning_rate=0.1, epochs=1000, momentum=False, batch_size=32, **kwargs):
        """Initialize the optimizer
        
            Args:
                learning_rate: learning rate
                epochs: number of epochs
                momentum: momentum
                batch_size: batch size (setting to n is equivalent to full batch)
                **kwargs: additional parameters
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, momentum=momentum, batch_size=batch_size, **kwargs)
        self.epsilon = kwargs.get("epsilon", 1e-8)

    def fit(self, model, X, y):
        """Fit the model to the data

            Args:
                model: model object
                X: design matrix (n x degree+1) matrix
                y: output data (n x 1) vector

            Returns:
                gradients_history: history of the gradients
                mse_history: history of the mean squared errors
        """
        n, _ = X.shape  # len of x
        m = int(n / self.batch_size)  # number of mini-batches
        model.theta = np.random.randn(X.shape[1],1)  # random initialization of the parameters
        gradients_history = np.zeros((self.epochs, X.shape[1]))  # history of the gradients
        mse_history = np.zeros(self.epochs)  # history of the mean squared errors
        change = 0.0
        
        for epoch in range(self.epochs):
            batch_gradient = np.zeros((m, X.shape[1]))  # store the gradients for each mini-batch
            self.giter = 0.0
            for i in range(m):
                random_index = self.batch_size*np.random.randint(m)
                X_mini = X[random_index:random_index+self.batch_size]
                y_mini = y[random_index:random_index+self.batch_size]

                regularization = 0 if model.reg_method == "ols" else 2*model.ridge_lambda*model.theta
                gradients = 2/self.batch_size * X_mini.T @ (X_mini @ model.theta - y_mini) + regularization

                self.giter += gradients * gradients
                self.adjusted_learning_rate = self.learning_rate / (np.sqrt(self.giter) + self.epsilon)  # Update learning rate

                if self.momentum:
                    new_change = self.adjusted_learning_rate * gradients + self.momentum_delta * change
                    model.theta -= new_change
                    change = new_change
                else:
                    model.theta -= self.adjusted_learning_rate * gradients

                # batch_gradient[i] = gradients.flatten()

            # gradients_history[epoch] = np.mean(batch_gradient, axis=0)
            mse_history[epoch] = self._mse(X, y, model)

        return gradients_history, mse_history
    
class RMSPropOptimizer(SGDOptimizer):
    def __init__(self, learning_rate=0.1, epochs=1000, momentum=False, batch_size=32, **kwargs):
        super().__init__(learning_rate=learning_rate, epochs=epochs, momentum=momentum, batch_size=batch_size, **kwargs)
        self.rho = kwargs.get("rho", 0.9)
        self.epsilon = kwargs.get("epsilon", 1e-8)

    def fit(self, model, X, y):
        n, _ = X.shape  # len of x
        m = int(n / self.batch_size)  # number of mini-batches
        model.theta = np.random.randn(X.shape[1],1)  # random initialization of the parameters
        gradients_history = np.zeros((self.epochs, X.shape[1]))  # history of the gradients
        mse_history = np.zeros(self.epochs)  # history of the mean squared errors
        change = 0.0
        giter = 0.0
        
        for epoch in range(self.epochs):
            batch_gradient = np.zeros((m, X.shape[1]))  # store the gradients for each mini-batch
            giter = 0.0
            for i in range(m):
                random_index = self.batch_size*np.random.randint(m)
                X_mini = X[random_index:random_index+self.batch_size]
                y_mini = y[random_index:random_index+self.batch_size]

                regularization = 0 if model.reg_method == "ols" else 2*model.ridge_lambda*model.theta
                gradients = 2/self.batch_size * X_mini.T @ (X_mini @ model.theta - y_mini) + regularization

                giter = self.rho * giter + (1 - self.rho) * gradients * gradients
                self.adjusted_learning_rate = self.learning_rate / (np.sqrt(giter) + self.epsilon)  # Update learning rate

                if self.momentum:
                    new_change = self.adjusted_learning_rate * gradients + self.momentum_delta * change
                    model.theta -= new_change
                    change = new_change
                else:
                    model.theta -= self.adjusted_learning_rate * gradients

                batch_gradient[i] = gradients.flatten()

            gradients_history[epoch] = np.mean(batch_gradient, axis=0)
            mse_history[epoch] = self._mse(X, y, model)

        return gradients_history, mse_history
    
class AdamOptimizer(SGDOptimizer):
    def __init__(self, learning_rate=0.1, epochs=1000, momentum=False, batch_size=32, **kwargs):
        super().__init__(learning_rate=learning_rate, epochs=epochs, momentum=momentum, batch_size=batch_size, **kwargs)
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.epsilon = kwargs.get("epsilon", 1e-8)

    def fit(self, model, X, y):
        n, _ = X.shape  # len of x
        m = int(n / self.batch_size)  # number of mini-batches
        model.theta = np.random.randn(X.shape[1],1)  # random initialization of the parameters
        gradients_history = np.zeros((self.epochs, X.shape[1]))  # history of the gradients
        mse_history = np.zeros(self.epochs)  # history of the mean squared errors
        change = 0.0
        miter = 0.0
        viter = 0.0
        
        for epoch in range(self.epochs):
            batch_gradient = np.zeros((m, X.shape[1]))  # store the gradients for each mini-batch
            miter = 0.0
            viter = 0.0
            for i in range(m):
                random_index = self.batch_size*np.random.randint(m)
                X_mini = X[random_index:random_index+self.batch_size]
                y_mini = y[random_index:random_index+self.batch_size]

                regularization = 0 if model.reg_method == "ols" else 2*model.ridge_lambda*model.theta
                gradients = 2/self.batch_size * X_mini.T @ (X_mini @ model.theta - y_mini) + regularization

                miter = self.beta1 * miter + (1 - self.beta1) * gradients
                viter = self.beta2 * viter + (1 - self.beta2) * gradients**2
                miter_hat = miter / (1 - self.beta1**(epoch*m + i + 1))
                viter_hat = viter / (1 - self.beta2**(epoch*m + i + 1))
                self.adjusted_learning_rate = self.learning_rate / (np.sqrt(viter_hat) + self.epsilon)  # Update learning rate

                if self.momentum:
                    new_change = self.adjusted_learning_rate * miter_hat + self.momentum_delta * change
                    model.theta -= new_change
                    change = new_change
                else:
                    model.theta -= self.adjusted_learning_rate * miter_hat

                batch_gradient[i] = gradients.flatten()

            gradients_history[epoch] = np.mean(batch_gradient, axis=0)
            mse_history[epoch] = self._mse(X, y, model)

        return gradients_history, mse_history