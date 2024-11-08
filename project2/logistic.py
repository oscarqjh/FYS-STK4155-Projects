import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01,epochs=1000, batch_size=32, l2_lambda=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None
        self.losses = []
        self.accuracy = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _logistic_cost_function(self, y_pred, y_true):
        """Binary cross-entropy loss with L2 regularization"""
        epsilon = 1e-12  # For numerical stability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add L2 regularization term
        l2_penalty = 0.5 * self.l2_lambda * np.sum(self.weights ** 2)
        return loss + l2_penalty

    def _compute_gradients(self, X, y, y_pred):
        """Compute gradients for weights and bias, including L2 regularization term"""
        n_samples = X.shape[0]
        
        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.l2_lambda * self.weights
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        return dw, db

    def fit(self, X, y):
        """Fit the logistic regression model to the data"""
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Forward pass: calculate predictions
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(linear_model)

                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Compute loss for monitoring
            total_loss = self._logistic_cost_function(self.sigmoid(np.dot(X, self.weights) + self.bias), y)
            self.losses.append(total_loss)

            # Compute accuracy
            acc = self.score(X, y)
            self.accuracy.append(acc)

    def predict_proba(self, X):
        """Return the probability estimates for the positive class"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predict binary class labels for samples in X"""
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def _accuracy_score(self, y_true, y_pred):
        """Compute the accuracy score"""
        return np.mean(y_true == y_pred) / len(y_true)
