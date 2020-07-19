import numpy as np

# la funzione costo nel forward mi deve restituire un numero
# nel backward mi deve restituire un vettore
class Cost_MSE():
    def forward(self, y_pred, y_real):
        n = y_real.shape[0]
        return (1/n) * np.sum((y_pred - y_real)**2)  # la somma dell'errore di tutti i neuroni k di output
        # uso axis=0 nel caso di categorical loss
    def backward(self, y_pred, y_real):
        m = y_real.shape[0]
        return (2/m) * np.sum(y_pred - y_real, axis=0)

# categorical cross entropy is when cross entropy is used together with softmax
# binary cross entropy is when cross entropy is used together with sigmoid
class Categorical_CrossEntropyLoss():
    def forward(self, y_pred, y_real):
        predictions = np.copy(y_pred)
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12) # avoid zero values for log
        n = y_real.shape[0]
        return - np.sum(y_real * np.log(y_pred))
    def backward(self, y_pred, y_real):
        return y_real - y_pred
        pass

class Binary_CrossEntropyLoss():
    def forward(self, y_pred, y_real):
        return - (y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
    def backward(self, y_pred, y_real):
        pass