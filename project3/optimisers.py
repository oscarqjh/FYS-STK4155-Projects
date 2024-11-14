import numpy as np

class Optimiser:
    """Abstract class for optimisers"""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_change(self, gradient):
        raise NotImplementedError
    
    def reset(self):
        pass
    
class Momentum:
    def __init__(self, beta):
        self.beta = beta
        self.velocity = 0

    def update_velocity(self, gradient):
        self.velocity = self.beta * self.velocity + gradient
        return self.velocity
    
    def reset_velocity(self):
        self.velocity = 0
    
class BasicOptimiser(Optimiser):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
    
    def update_change(self, gradient):
        return self.learning_rate * gradient
    
class BasicMomentumOptimiser(BasicOptimiser, Momentum):
    def __init__(self, learning_rate, beta):
        BasicOptimiser.__init__(self, learning_rate)
        Momentum.__init__(self, beta)
    
    def update_change(self, gradient):
        base_change = BasicOptimiser.update_change(self, gradient)
        return self.update_velocity(base_change)
    
    def reset(self):
        self.reset_velocity()

class AdagradOptimiser(Optimiser):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G_t = None

    def update_change(self, gradient):
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[1]))
        self.G_t += gradient @ gradient.T

        G_t_inv = 1 / (self.epsilon + np.sqrt(np.reshape(np.diag(self.G_t), (self.G_t.shape[0], 1))))
        return self.learning_rate * G_t_inv * gradient

    def reset(self):
        self.G_t = None

class AdagradMomentumOptimiser(AdagradOptimiser, Momentum):
    def __init__(self, learning_rate, beta, epsilon=1e-8):
        AdagradOptimiser.__init__(self, learning_rate, epsilon)
        Momentum.__init__(self, beta)
    
    def update_change(self, gradient):
        base_change = AdagradOptimiser.update_change(self, gradient)
        return self.update_velocity(base_change)
    
    def reset(self):
        super().reset()
        self.reset_velocity()

class RMSPropOptimiser(Optimiser):
    def __init__(self, learning_rate, rho, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.second = 0.0

    def update_change(self, gradient):
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.learning_rate * gradient / (np.sqrt(self.second) + self.epsilon)
    
    def reset(self):
        self.second = 0.0

class RMSPropMomentumOptimiser(RMSPropOptimiser, Momentum):
    def __init__(self, learning_rate, rho, beta, epsilon=1e-8):
        RMSPropOptimiser.__init__(self, learning_rate, rho, epsilon)
        Momentum.__init__(self, beta)
    
    def update_change(self, gradient):
        base_change = RMSPropOptimiser.update_change(self, gradient)
        return self.update_velocity(base_change)
    
    def reset(self):
        super().reset()
        self.reset_velocity()

class AdamOptimiser(Optimiser):
    def __init__(self, learning_rate, beta1, beta2, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0.0
        self.v = 0.0
        self.t = 1

    def update_change(self, gradient):
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient * gradient
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        self.m = 0.0
        self.v = 0.0
        self.t += 1

class AdamMomentumOptimiser(AdamOptimiser, Momentum):
    def __init__(self, learning_rate, beta1, beta2, beta, epsilon=1e-8):
        AdamOptimiser.__init__(self, learning_rate, beta1, beta2, epsilon)
        Momentum.__init__(self, beta)
    
    def update_change(self, gradient):
        base_change = AdamOptimiser.update_change(self, gradient)
        return self.update_velocity(base_change)
    
    def reset(self):
        super().reset()
        self.reset_velocity()