import numpy as np                  # for basic operations over arrays
from scipy.spatial import distance  # to compute the Gaussian kernel
import cvxopt                       # to solve the dual optimization problem
import copy                         # to copy numpy arrays 


class SVM:
    linear = lambda x, x_prime , c=0: x @ x_prime .T
    polynomial = lambda x, x_prime , Q=5: (1 + x @ x_prime.T)**Q
    rbf = lambda x, x_prime , gamma=10: np.exp(-gamma * distance.cdist(x, x_prime,'sqeuclidean'))
    kernel_funs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}
    
    def __init__(self, kernel='rbf', C=1, k=2):
        # set the hyperparameters
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funs[kernel]
        self.C = C                  # regularization parameter
        self.k = k                  # kernel parameter
        
        # training data and support vectors
        self.X, y = None, None
        self.alphas = None
        
        # for multi-class classification
        self.multiclass = False
        self.clfs = []                                  

    def fit(self, X, y, eval_train=False):
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)
        
        # relabel if needed
        if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
        # ensure y has dimensions Nx1
        self.y = y.reshape(-1, 1).astype(np.double) # Has to be a column vector
        self.X = X
        N = X.shape[0]
        
        # compute the kernel over all possible pairs of (x, x') in the data
        self.K = self.kernel(X, X, self.k)
        
        # For 1/2 x^T P x + q^T x
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        
        # For Ax = b
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))

        # For Gx <= h
        G = cvxopt.matrix(np.vstack((-np.identity(N),
                                    np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N,1)),
                                    np.ones((N,1)) * self.C)))

        # Solve    
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])
            
        # Maps into support vectors
        self.is_sv = ((self.alphas > 1e-3) & (self.alphas <= self.C)).squeeze()
        self.margin_sv = np.argmax((1e-3 < self.alphas) & (self.alphas < self.C - 1e-3))
        
        if eval_train:  
            print(f"Finished training with accuracy {self.evaluate(X, y)}")


    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))      # number of classes
        # for each pair of classes
        for i in range(self.k):
            # get the data for the pair
            Xs, Ys = X, copy.copy(y)
            # change the labels to -1 and 1
            Ys[Ys!=i], Ys[Ys==i] = -1, +1
            # fit the classifier
            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            # save the classifier
            self.clfs.append(clf)
        if eval_train:  
            print(f"Finished training with accuracy {self.evaluate(X, y)}")

    def predict(self, X_t):
        if self.multiclass: return self.multi_predict(X_t)
        xₛ, yₛ = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        alphas, y, X= self.alphas[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

        b = yₛ - np.sum(alphas * y * self.kernel(X, xₛ, self.k), axis=0)
        score = np.sum(alphas * y * self.kernel(X, X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score

    def multi_predict(self, X):
        # get the predictions from all classifiers
        preds = np.zeros((X.shape[0], self.k))
        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)
        
        # get the argmax and the corresponding score
        return np.argmax(preds, axis=1), np.max(preds, axis=1)

    def evaluate(self, X,y):  
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)
