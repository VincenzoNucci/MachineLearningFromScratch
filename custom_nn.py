import numpy as np 
import matplotlib.pyplot as plt
from copy import copy
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from activation_functions import *
from cost_functions import *

np.random.seed(1)

def mse(y_pred, y_real):
    return np.sum((y_pred - y_real)**2) / 2

class Model():
    pass
class Layer():
    pass

# la distribuzione uniforme si è comportata meglio della distribuzione N(0,1)
class Neuron():
    def __init__(self, n_outputs, activation='relu'):
        # qui ho scritto n_inputs ma un neurone ha un solo input
        self.weights = np.random.uniform(low=0.01, high=0.05, size=(1,n_outputs)) # vettore dei pesi di un singolo neurone
        self.bias = 1 # bias del singolo neurone
        if activation == 'softmax':
            self.activation = Activation_Softmax()
        elif activation == 'sigmoid':
            self.activation = Activation_Sigmoid()
        elif activation == 'tanh':
            self.activation = Activation_Tanh()
        else:
            self.activation = Activation_ReLU()
    def forward(self, input):
        self.z = np.dot(input, self.weights) + self.bias # z output del neurone
        self.a = self.activation.forward(self.z) # a output attivato del neurone
        return self.a
    def backward(self, X, y, output, step):
        pass
    def __repr__(self):
        return str(self.weights)
    

class Dense(Layer):
    def __init__(self, input_shape, name=None, activation='relu'):
        self.name = name
        self.is_output = False
        self.weights = np.random.uniform(low=0.01, high=0.10, size=input_shape) # vettore dei pesi di un singolo neurone
        self.biases = np.ones((1,input_shape[1])) # bias di ogni singolo neurone
        if activation == 'softmax':
            self.activation = Activation_Softmax()
        elif activation == 'sigmoid':
            self.activation = Activation_Sigmoid()
        elif activation == 'tanh':
            self.activation = Activation_Tanh()
        elif activation == 'identity':
            self.activation = Activation_Identity()
        else: #activation == 'relu':
            self.activation = Activation_ReLU()
        self.cost = Cost_MSE()
    def set_as_output(self, is_output=True):
        self.is_output = is_output
    def forward(self, inputs, debug=False, epsilon=None):
        self.net_input = inputs
        if debug:
            augmented_parameters = np.zeros(epsilon.shape)
            weights_column_vector = np.reshape(self.weights,(-1,1))
            biases_column_vector = np.reshape(self.biases,(-1,1))
            concatenated_parameters = np.concatenate((weights_column_vector, biases_column_vector))
            for i in range(concatenated_parameters.shape[0]):
                augmented_parameters[i] = concatenated_parameters[i]
            # make the augmented parameter long as theta in order to sum them
            # this because epsilon is a standard basis vector
            augmented_parameters += epsilon
            # rebuild the weights matrix and biases vector to apply forward propagation
            weights_end = self.weights.shape[0] * self.weights.shape[1]
            biases_end = self.biases.shape[0] * self.biases.shape[1] + weights_end
            weights = np.reshape(augmented_parameters[0:weights_end],self.weights.shape)
            biases = np.reshape(augmented_parameters[weights_end:biases_end], self.biases.shape)
            output = np.dot(inputs, weights) + biases
            activated_output = self.activation.forward(output)
            return activated_output
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activated_output = self.activation.forward(self.output)
        return self.activated_output
    def backward(self, X, y, output, step):
        if self.is_output:
            errore = self.cost.backward(output, y) #(a_k - y_hat_k) k - 120x3 oppure 120x3 - 120x3 [n_features x n_classlabels]
            #print('errore shape', errore.shape)
            #output_error = self.cost.backward(output, y)
            #print('derivata output shape', self.activation.backward(self.output).shape)
            #print('sigmoid prime',np.sum(self.activation.backward(self.output)))
            delta_k = self.activation.backward(self.output)* errore # 120x3 o 120x3 = 120x3 dimensione uguale ai pesi del layer di output perchè 5 sono i neuroni di input del layer precedente e 3 sono i neuroni di output che sono le classlabels
            # delta_k quanto il neurone k si discosta dal valore reale
            #print('delta_k shape',delta_k.shape)
            # per calcolare il gradiente moltiplico l'output attivato del neurone precedente che sarebbe l'input del neurone attuale con il delta_k 5x3
            #print('shape net input', self.net_input.shape)
            grad = np.dot(self.net_input.T, delta_k) # 5x120 o 120x3 = 5x3
            # gradient check

            #print('shape pesi k', self.weights.shape)
            #update pesi
            self.grad_w = grad 
            #print(grad_w)
            self.grad_b = np.sum(delta_k * 1,axis=0) # diventa vettore (1,3)
            #print(np.sum(self.weights))
            self.weights -= step * self.grad_w 
            self.biases -= step * self.grad_b
            #print(np.sum(self.weights))
            return np.dot(delta_k ,self.weights.T) # 120x3 o 3x5 = 120x5
        else:
            delta_j = self.activation.backward(self.output) * output
            #print('delta_j', np.sum(delta_j))
            grad = np.dot(self.net_input.T, delta_j)
            self.grad_w = grad
            self.grad_b = np.sum(delta_j * 1, axis=0)
            #print(np.sum(self.weights))
            self.weights -= step * self.grad_w
            self.biases -= step * self.grad_b
            #print(np.sum(self.weights))
            return np.dot(delta_j, self.weights.T)
    def compile(self, optimizer=None, loss='mse'):
        if loss == 'categorical_crossentropy':
            self.cost = Categorical_CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.cost = Binary_CrossEntropyLoss()
        else:
            self.cost = Cost_MSE()
    def summary(self):
        return '''{}
        {}'''.format(self.name, self.weights.shape)
    def get_parameters(self):
        return self.weights, self.biases
    def get_gradients(self):
        return self.grad_w, self.grad_b

class NeuralNet():
    def __init__(self):
        self.layers = []
        self.layers_output = []
        self.cost = None
    def add(self,layer):
        self.layers.append(layer)
    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
    def forward(self, inputs, debug=False, epsilon=None):
        input = np.copy(inputs)
        for layer in self.layers:
            output = layer.forward(input, debug=debug, epsilon=epsilon)
            input = output
        return input
    def backward(self, X, y, output, step):
        prev_delta = None
        out = output
        for layer in self.layers[::-1]:
            prev_delta = layer.backward(X, y, out, step)
            out = prev_delta
    def fit(self, X, y, batch_size=1, epochs=10, step=0.05, shuffle=True):
        self.layers[-1].set_as_output()
        self.error = []
        i = 0.005 * epochs
        for epoch in range(epochs):
            #print('X shape in fit',X.shape)W
            if shuffle:
                np.random.shuffle(X)
            batches = int(np.ceil(X.shape[0]/batch_size))
            batches_error = []
            for t in range(batches):
                batch_X = X[t*batch_size:np.min([X.shape[0],(t+1)*batch_size]),:]
                batch_y = y[t*batch_size:np.min([y.shape[0],(t+1)*batch_size]),:]
                output = self.forward(batch_X)
                #print('output shape', output.shape)
                # output è l'uscita attivata del neurone di output layer ed è grande quanto le y reali e va bene
                #print('output neuron ou',np.sum(output))
                cost = self.cost.forward(output,batch_y)
                # cost è uno scalare e va bene
                # cost = mean_squared_error(y,output)
                batches_error.append(cost)
                
                self.backward(batch_X, batch_y, output, step)
            self.error.append(np.mean(batches_error))
            if epoch % i == 0:
                    print('epoch:', epoch, 'error:', np.mean(self.error))
        plt.plot(np.arange(epochs),self.error)
        plt.show()
        return self
    def predict(self, X):
        return self.forward(X)
    def summary(self):
        for layer in self.layers:
            print(layer.summary())
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print(y_pred)
        print(confusion_matrix(y,y_pred))
    def compile(self, loss='mse'):
        if loss == 'categorical_crossentropy':
            self.cost = Categorical_CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.cost = Binary_CrossEntropyLoss()
        else:
            self.cost = Cost_MSE()
        for layer in self.layers:
            layer.compile(loss=loss)
    def parameters_to_theta(self):
        keys = []
        count = 0
        # inizializzo theta un vettore grande quanti i pesi del primo layer
        theta = np.empty((self.layers[0].get_parameters()[0].shape[0],1))
        theta = []
        for layer in self.layers:
            w, b = layer.get_parameters()
            #flatten parameter w
            new_vector = np.reshape(w, (-1,1))
            #theta = np.concatenate((theta, new_vector), axis=0)
            theta.append(new_vector)


            #flatten parameter b
            new_vector = np.reshape(b, (-1,1))
            #theta = np.concatenate((theta, new_vector), axis=0)
            theta.append(new_vector)
        return np.vstack(theta) # un singolo vettore con tutti i parametri della rete
    def gradients_to_theta(self):
        theta = []
        for layer in self.layers:
            grad_w, grad_b = layer.get_gradients()
            new_vector = np.reshape(grad_w, (-1,1))
            theta.append(new_vector)

            new_vector = np.reshape(grad_b, (-1,1))
            theta.append(new_vector)
        return np.vstack(theta)
    def gradient_check(self, X, y, epsilon=1e-7):
        theta = self.parameters_to_theta()
        dtheta = self.gradients_to_theta() # 73x1 vettore riga di 73 gradienti (i pesi complessivi + i biases)
        # X 120x4
        # y 120x4
        num_parameters = theta.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        dtheta_approx = np.zeros((num_parameters, 1))
        for i in range(num_parameters):
            theta_plus = np.zeros((num_parameters,1))
            theta_plus[i] = epsilon # vettore riga 73x1 con ognuno dei 73 valori aumentato di epsilon
            # ora devo calcolare il costo con i parametri aumentati di un epsilon
            J_plus[i] = self.cost.forward(self.forward(X, debug=True, epsilon=theta_plus),y)
            # J_plus è il costo con in parametri aumentati di un epsilon
            theta_minus = np.zeros((num_parameters,1))
            theta_minus[i] = - epsilon
            J_minus[i] = self.cost.forward(self.forward(X, debug=True, epsilon=theta_minus),y)
            # J_minus è il costo con i parametri aumentati di un epsilon
            # dtheta_approx deve essere un vettore colonna 73x1
            dtheta_approx[i] = (J_plus[i] - J_minus[i])/ (2 * epsilon)
            
        numerator = np.linalg.norm(dtheta - dtheta_approx)
        denominator = np.linalg.norm(dtheta_approx) + np.linalg.norm(dtheta)
        difference = numerator / denominator
        return difference



X, y = load_iris(return_X_y=True)

X = (X - np.mean(X)) / np.std(X) # standardize the data to improve network convergence
# now mean is close to zero
# now std is close to 1 or 1.0
y = y.reshape((-1,1)) # iris dataset has (150,3) output but we need (150,)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('training samples', X_train.shape)
model = NeuralNet()

output_layer = Dense((10,3),name='output_layer',activation='sigmoid')

model.add(Dense((4,10),name='input_layer',activation='relu'))
model.add(Dense((10,10),name='hidden_layer',activation='relu'))
model.add(output_layer)

model.compile(loss='categorical_crossentropy')
#X_train = np.random.randn(1,120).T
#y_train = np.greater_equal(X_train,0.5).astype(int).T
#model.forward(X_train)
# con eta = 0.1 la rete diverge (i pesi diventano troppo grandi) perchè quando scende nel gradiente lo fa con passi troppo lunghi

# alleno la rete con pochi input giusto per calcolare la differenza di gradiente

model.fit(X_train,y_train, batch_size=5, epochs=1000, step=1e-4)
difference = model.gradient_check(X_train, y_train)
if difference <= 1e-7:
    print('backpropagation works')
else:
    print(difference, 'not gud')

print(model.predict(X_test[0]))
#print(theta)

#print(model.summary())
#first_pass_out = model.forward(X_train)
#print(first_pass_out)
#model.backward(X_train, y_train, first_pass_out, eta)
#second_pass_out = model.forward(X_train)
#print(second_pass_out)
#model.evaluate(X_test, y_test)
# predict
# net_predict = activation3.forward(layer3.forward(activation2.forward(layer2.forward(activation1.forward(layer1.forward(X))))))
#loss = mse(net_predict,y)
#gradient = activation3.backward
#theta =  eta * gradient(net_predict)
#print(loss)