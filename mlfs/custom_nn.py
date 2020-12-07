import numpy as np 
import matplotlib.pyplot as plt
from copy import copy
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from activation_functions import *
from loss_functions import *
from preprocessing_functions import *

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
        self.weights = np.random.uniform(low=0.01, high=0.10, size=(1,n_outputs)) # vettore dei pesi di un singolo neurone
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
    def __init__(self, name=None, activation='relu', regularization='l2', input_shape=None):
        # ADAM
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_t = 0
        self.v_t = 0
        ##################
        self.name = name
        self.is_output = False
        self.weights = np.random.uniform(size=input_shape) # vettore dei pesi di un singolo neurone
        self.biases = np.ones((1,input_shape[1])) # bias di ogni singolo neurone
        if activation == 'softmax':
            self.activation = Activation_Softmax()
        elif activation == 'softplus':
            self.activation = Activation_SoftPlus()
        elif activation == 'sigmoid':
            self.activation = Activation_Sigmoid()
        elif activation == 'tanh':
            self.activation = Activation_Tanh()
        elif activation == 'identity':
            self.activation = Activation_Identity()
        elif activation == 'leaky_relu':
            self.activation = Activation_LeakyReLU()
        else: #activation == 'relu':
            self.activation = Activation_ReLU()
        self.cost = MeanSquaredErrorLoss()
    def set_as_output(self, is_output=True):
        self.is_output = is_output
    def forward(self, inputs, debug=False, epsilon=None, offset=0):
        self.net_input = np.copy(inputs)
        if debug:
            print('offset',offset)
            augmented_parameters = np.zeros(epsilon.shape)
            weights_column_vector = np.reshape(self.weights,(-1,1))
            biases_column_vector = np.reshape(self.biases,(-1,1))
            concatenated_parameters = np.concatenate((weights_column_vector, biases_column_vector))
            for i in range(concatenated_parameters.shape[0]):
                augmented_parameters[i+offset] = concatenated_parameters[i]
            # make the augmented parameter long as theta in order to sum them
            # this because epsilon is a standard basis vector
            augmented_parameters += epsilon
            print(augmented_parameters - epsilon)
            # rebuild the weights matrix and biases vector to apply forward propagation
            weights_end = offset + (self.weights.shape[0] * self.weights.shape[1])
            biases_end = offset + (self.biases.shape[0] * self.biases.shape[1]) + weights_end
            print(biases_end-weights_end)
            weights = np.reshape(augmented_parameters[offset:weights_end],self.weights.shape)
            biases = np.reshape(augmented_parameters[weights_end:biases_end], self.biases.shape)
            output = np.dot(inputs, weights) + biases
            activated_output = self.activation.forward(output)
            return activated_output
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activated_output = self.activation.forward(self.output)
        return self.activated_output
    def backward(self, X, y, output, t, step, l2=0.05):
        m = X.shape[0] # number of examples
        if self.is_output:
            errore = self.loss.backward(output, y) #(a_k - y_hat_k) k - 120x3 oppure 120x3 - 120x3 [n_features x n_classlabels]
            #print('errore shape', errore.shape)
            #output_error = self.cost.backward(output, y)
            #print('derivata output shape', self.activation.backward(self.output).shape)
            #print('sigmoid prime',np.sum(self.activation.backward(self.output)))
            delta_k = self.activation.backward(self.output)* errore # 120x3 o 120x3 = 120x3 dimensione uguale ai pesi del layer di output perchè 5 sono i neuroni di input del layer precedente e 3 sono i neuroni di output che sono le classlabels
            # delta_k quanto il neurone k si discosta dal valore reale
            #print('delta_k shape',delta_k.shape)
            # per calcolare il gradiente moltiplico l'output attivato del neurone precedente che sarebbe l'input del neurone attuale con il delta_k 5x3
            #print('shape net input', self.net_input.shape)
            # net input al neurone di ouput è la funzione di attivazione del layer precedente
            g_t = np.dot(self.net_input.T, delta_k) # 5x120 o 120x3 = 5x3
            g_t += (l2/m)*self.weights
            # gradient check
            # ADAM
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * g_t
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (g_t ** 2)
            lr_t = step * (np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t))
            m_hat = self.m_t / (1 - (self.beta1 ** t))
            v_hat = self.v_t / (1 - (self.beta2 ** t))
            
            # self.weights += return
            # - lr_t * (m_hat / (np.sqrt(v_hat)+self.epsilon))
            #print('shape pesi k', self.weights.shape)
            #update pesi con l2 regularization
            #self.grad_w = g_t + (l2 / m)*self.weights
            #print(self.name,'grad_w',self.grad_w)
            self.grad_b = np.sum(delta_k * 1,axis=0) # diventa vettore (1,3)
            #print(np.sum(self.weights))
            self.weights -= lr_t * (m_hat / (np.sqrt(v_hat)+self.epsilon))
            self.biases -= step * self.grad_b
            #print(np.sum(self.weights))
            return np.dot(delta_k ,self.weights.T) # 120x3 o 3x5 = 120x5
        else:
            delta_j = self.activation.backward(self.output) * output
            #print('delta_j', np.sum(delta_j))
            g_t = np.dot(self.net_input.T, delta_j)
            # grad = self.optimizer.get_grad(self.net_input, delta_j, t, step)
            g_t += (l2/m)*self.weights
            # ADAM
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * g_t
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (g_t * g_t)
            lr_t = step * (np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t))
            m_hat = self.m_t / (1 - (self.beta1 ** t))
            v_hat = self.v_t / (1 - (self.beta2 ** t))
            ########
            # self.grad_w = grad + (l2 / m) * self.weights
            self.grad_b = np.sum(delta_j * 1, axis=0)
            #print(np.sum(self.weights))
            self.weights -= lr_t * (m_hat / (np.sqrt(v_hat)+self.epsilon))
            self.biases -= step * self.grad_b
            #print(np.sum(self.weights))
            return np.dot(delta_j, self.weights.T)
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
    def summary(self):
        return '''{}
        {}'''.format(self.name, self.weights)
    def get_parameters(self):
        return self.weights, self.biases
    def get_gradients(self):
        return self.grad_w, self.grad_b

class NeuralNet():
    def __init__(self, layers=None):
        if layers != None:
            for layer in layers:
                self.add(layer)
        else:
            self.layers = []
        # self.layers_output = []
        self.loss = None
        self.optimizer = None
    def add(self,layer):
        self.layers.append(layer)
    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
    def forward(self, inputs, verbose=0):
        input = np.copy(inputs)
        for layer in self.layers:
            output = layer.forward(input, verbose=0)
            w, b = layer.get_parameters()
            input = output
        return input # is now total layer output (activated)
    def backward(self, X, y, output, epoch, step):
        prev_delta = None
        out = np.copy(output)
        for layer in self.layers[::-1]:
            prev_delta = layer.backward(X, y, out, epoch, step)
            out = prev_delta
    def fit(self, X, y, batch_size=32, epochs=10, shuffle=True):
        self.layers[-1].set_as_output()
        self.error = []
        
        for epoch in range(epochs):
            #print('X shape in fit',X.shape)W
            if shuffle:
                # np.random.shuffle(X) shuffle does not copy array
                # permutation = np.random.permutation(X.shape[0])
                
            batches = int(np.ceil(X.shape[0]/batch_size))
            batches_error = []
            for t in range(batches):
                batch_X = _X[t*batch_size:np.min([X.shape[0],(t+1)*batch_size]),:]
                batch_y = y[t*batch_size:np.min([y.shape[0],(t+1)*batch_size]),:]
                # print('batch data',batch_X)
                output = self.forward(batch_X)
                # print('output shape', output.shape)
                # output è l'uscita attivata del neurone di output layer ed è grande quanto le y reali e va bene
                #print('output neuron ou',np.sum(output))
                cost = self.loss.forward(output,batch_y)
                cost += self.regularization.forward(X, self.layers)
                # cost è uno scalare e va bene
                # cost = mean_squared_error(y,output)
                batches_error.append(cost)
                print('batch: {0}/{1}, epoch: {2}, error: {3}'.format(t+1,batches,epoch+1,np.mean(batches_error)))
                
                self.backward(batch_X, batch_y, output, epoch+1, step)
            self.error.append(np.mean(batches_error))
            # if epoch % i == 0: # per adesso me le stampi tutte
            # print('batch: {0}/{1}, epoch: {2}, error: {3}'.format(t+1,batches,epoch,np.mean(self.error)))
            # self.summary()
        # plt.plot(np.arange(epochs),self.error)
        # plt.show()
        return self
    def predict(self, X):
        y_pred = self.forward(X)
        return y_pred
    def summary(self):
        for layer in self.layers:
            print(layer.summary())
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print(y_pred)
        print(confusion_matrix(y,y_pred))
    def compile(self, optimizer='sdg', loss='mse'):
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            if optimizer == 'adam':
                self.optimizer = Adam()
        if loss == 'categorical_crossentropy':
            self.loss = Categorical_CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.loss = Binary_CrossEntropyLoss()
        else:
            self.loss = MeanSquaredErrorLoss()
        for layer in self.layers:
            layer.compile(optimizer=self.optimizer, loss=self.loss)
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
        dtheta = self.gradients_to_theta() # 73x1 vettore colonna di 73 gradienti (i pesi complessivi + i biases)
        # X 120x4
        # y 120x4
        num_parameters = theta.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        dtheta_approx = np.zeros((num_parameters, 1))
        for i in range(num_parameters):
            theta_plus = np.zeros((num_parameters,1))
            theta_plus[i] = epsilon # vettore colonna 73x1 con ognuno dei 73 valori aumentato di epsilon
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

# X = StandardScaler().fit(X) # standardize the data to improve network convergence
# now mean is close to zero
# now std is close to 1 or 1.0
# X = Normalizer().fit(X) # normalize X (all values between 0 and 1)
y = y.reshape((-1,1)) # iris dataset has (150,3) output but we need (150,)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('training samples', X_train.shape)
model = NeuralNet([
    Dense(4,activation='relu',input_shape=(4,),use_bias=False),
    Dense(2,activation='relu',use_bias=False),
    Dense(5,activation='relu',use_bias=False),
    Dense(1,activation='sigmoid',use_bias=False)
])

optimizer = SGD(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy')
#X_train = np.random.randn(1,120).T
#y_train = np.greater_equal(X_train,0.5).astype(int).T
#model.forward(X_train)
# con eta = 0.1 la rete diverge (i pesi diventano troppo grandi) perchè quando scende nel gradiente lo fa con passi troppo lunghi

# alleno la rete con pochi input giusto per calcolare la differenza di gradiente

model.fit(X_train,y_train, batch_size=5, epochs=2, step=0.001)
# difference = model.gradient_check(X_train, y_train)
# if difference <= 1e-7:
#     print('backpropagation works')
# else:
#     print(difference, 'not gud')


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