import numpy as np
from scipy.special import expit as sigmoid

class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)
    def backward(self, inputs):
        return np.greater(inputs,0).astype(int) # altrimenti ritorna True o False

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

class Activation_Softmax:
    def forward(self, inputs):
        return np.exp(inputs) / np.sum(np.exp(inputs),axis=0)
    def backward(self, inputs):
        #Sz = np.reshape(inputs, (-1,1))
        Sz = inputs
        D = -np.outer(Sz, Sz) + np.diagflat(Sz)
        return D

class Activation_StableSoftmax:
    # used to avoid NaNs
    def forward(self, inputs):
        shift = inputs - np.max(inputs)
        return np.exp(shift) / np.sum(np.exp(shift))
    def backward(self, inputs):
        raise NotImplementedError

class Activation_Sigmoid:
    def sigmoid_throws_runtime_error(self,s):
        return 1 / (1 + np.exp(-s))
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