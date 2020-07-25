import numpy as np
from scipy.special import expit as sigmoid

class Activation_ReLU: # REctified Linear Unit
    def forward(self, inputs):
        return np.maximum(0, inputs)
    def backward(self, inputs):
        return np.greater(inputs,0).astype(int) # altrimenti ritorna True o False

class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, inputs):
        D = np.copy(inputs)
        D[D < 0] *= self.alpha
        return D
    def backward(self, inputs):
        D = np.ones_like(inputs)
        D[D < 0] = self.alpha
        return D

#TODO da controllare
class Activation_ELU: # Exponential Linear Unit
    '''
    Parameter alpha can range between 0 and 1.
    '''
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    def ELU(self, inputs):
        if np.all(inputs) <= 0:
            return self.alpha * (np.exp(inputs) - 1)
        else:
            return inputs
    def forward(self, inputs):
        return self.ELU(inputs)
    def backward(self,inputs):
        if np.all(inputs) <= 0:
            return self.ELU(inputs) + self.alpha
        else:
            return 1

#TODO da controllare
class Activation_SELU: # Scaled Exponential Linear Unit
    def __init__(self, lambd=1.0507, alpha=1.67326):
        self.lambd = lambd
        self.alpha = alpha
    def forward(self,inputs):
        if np.all(inputs) < 0:
            return self.lambd * (self.alpha * (np.exp(inputs) - 1))
        else:
            return self.lambd * inputs
    def backward(self, inputs):
        if np.all(inputs) < 0:
            return self.lambd * (self.alpha * np.exp(inputs))
        else:
            return self.lambd * 1

class Activation_Identity:
    def forward(self, inputs):
        return inputs
    def backward(self, inputs):
        return 1

class Activation_Step:
    def forward(self, inputs):
        np.heaviside(inputs,1)
    def backward(self, inputs):
        raise NotImplementedError

#TODO da controllare
# returns only NaNs
class Activation_SoftPlus:
    def forward(self, inputs):
        logarg = (1 + np.exp(inputs))
        return np.log(logarg)
    def backward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class Activation_Softmax:
    def softmax(self, inputs):
        print('softmax inputs', inputs)
        e_x = np.exp(inputs - np.max(inputs))
        print('e_x',e_x)
        return e_x / np.sum(e_x,axis=0)
    def forward(self, inputs):
        return self.softmax(inputs)
    def backward(self, inputs):
        #Sz = np.reshape(inputs, (-1,1))
        Sz = inputs # 120x3
        D = - np.outer(Sz, Sz) + np.diagflat(Sz) # (1 - S_i) * S_j = -S_i*S_j + S_j
        return np.reshape(np.sum(D,axis=1),inputs.shape)
        # sum axis=0 mantiene le features ma cambia le righe (1,n_features)
        # sum axis=1 mantiene le righe ma cambia le features (n_samples, )

class Activation_StableSoftmax:
    # used to avoid NaNs
    def forward(self, inputs):
        shift = inputs - np.max(inputs)
        return np.exp(shift) / np.sum(np.exp(shift))
    def backward(self, inputs):
        raise NotImplementedError

class Activation_Sigmoid: # aka logistic function
    def sigmoid_throws_runtime_error(self,s):
        return 1.0 / (1 + np.exp(-s))
    def forward(self, inputs):
        return sigmoid(inputs)
    def backward(self, inputs):
        return sigmoid(inputs) * (1 - sigmoid(inputs))

class Activation_Tanh:
    def tanh(self, inputs):
        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    def forward(self, inputs):
        return self.tanh(inputs)
    def backward(self, inputs):
        return 1 - self.tanh(inputs)**2