import numpy as np

# la funzione costo nel forward mi deve restituire un numero
# nel backward mi deve restituire un vettore
class Cost_MSE():
    def forward(self, y_pred, y_real):
        n = y_real.shape[0]
        return (1/n) * np.sum((y_pred - y_real)**2)  # la somma dell'errore di tutti i neuroni k di output
    def backward(self, y_pred, y_real):
        m = y_real.shape[0]
        return (2/m) * np.sum(y_pred - y_real)

class CrossEntropyLoss():
    def forward(self, y_pred, y_real):
        return - np.sum(y_real * np.log(y_pred))
    def backward(self, y_pred, y_real):
        return 0