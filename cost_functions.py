import numpy as np

# la funzione costo nel forward mi deve restituire un numero
# nel backward mi deve restituire un vettore
class MeanSquaredErrorLoss():
    def forward(self, y_real, y_pred):
        n = y_real.shape[0]
        return (1/n) * np.sum((y_pred - y_real)**2)  # la somma dell'errore di tutti i neuroni k di output
        # uso axis=0 nel caso di categorical loss
    def backward(self, y_real, y_pred):
        m = y_real.shape[0]
        return -1 * (2/m) * np.sum((y_pred - y_real))

class RootMeanSquaredErrorLoss():
    def forward(self, y_real, y_pred):
        n = y_real.shape[0]
        return np.sqrt((1/n) * np.sum((y_pred - y_real)**2))
    def backward(self, y_real, y_pred):
        pass
        #TODO

# categorical cross entropy is when cross entropy is used together with softmax
# binary cross entropy is when cross entropy is used together with sigmoid
class Categorical_CrossEntropyLoss():
    def forward(self, y_pred, y_real):
        predictions = np.copy(y_pred)
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12) # avoid zero values for log
        n = y_real.shape[0]
        return - (1 / n) * np.sum(y_real * np.log(predictions),axis=0)
    def backward(self, y_pred, y_real):
        return y_real - y_pred # è corretta

class Binary_CrossEntropyLoss():
    def forward(self, y_pred, y_real):
        n = y_real.shape[0]
        return - (1 / n) * np.sum((y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)),axis=0)
    def backward(self, y_pred, y_real):
        return y_real - y_pred