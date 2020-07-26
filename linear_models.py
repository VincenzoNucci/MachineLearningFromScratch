import numpy as np
from scipy import stats
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from preprocessing_functions import StandardScaler

class LinearModel(object):
    def __init__(self):
        self.weights = None
        self.theta = None
        self.intercept_ = None
        self.coef_ = None
        self.n_samples = None
        self.n_features = None

    def sem(self, param): # standard error of the mean, aka standar error of the estimate
        return np.std(param,axis=0) / np.sqrt(self.theta.shape[0])
    def t_value_abs(self, param):
        return np.abs(param / self.sem(param))

class LinearRegression(LinearModel):
    def __init__(self):
        super().__init__()
        # self.weights è matrice Nxd+1 (compreso il bias)

    def fit(self, X, y, epochs=200, step=0.01):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # define matrix w with instances plus bias
        # self.theta = np.hstack((np.ones((X.shape[0], 1)),X))
        self.theta = np.random.uniform(size=(self.n_features + 1, 1))
        theta_estimate = np.copy(self.theta)
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        error = []
        for epoch in range(epochs):
            y_pred = _X @ self.theta
            # compute error with mse
            errore = (1/self.n_samples) * np.sum((y_pred - y)**2)
            error.append(errore)
            # compute gradient of cost
            grad = (2/self.n_samples) * (_X.T.dot(_X.dot(self.theta) - y))
            # update weights
            self.theta -= step * grad
            print('epoch: {}, error: {}'.format(epoch+1,np.mean(error)))
            
            #errore = self.cost.forward(self.predict(X),y)
            #print('epoch: {}, error: {}'.format(epoch+1, errore))
            #error.append(errore)
            #self.theta -= step * self.cost.backward(errore, y)
        
        # self.weights = np.hstack((np.ones((X.shape[0], 1)),X))
        # compute in a single step, the best parameters that fit the data
        # self.theta = np.linalg.inv(np.dot(self.weights.T,self.weights)).dot(self.weights.T).dot(y)
        # self.theta = np.reshape(self.theta, (-1,1)) # reshape into a 2d vector
        # print('theta best', self.theta.shape) (14,1)
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

        return self

    def predict(self, X):
        # if passed only a sample instead of test matrix
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return _X @ self.theta

    def summary(self):
        for w in self.theta:
            print(w, self.sem(w), self.t_value_abs(w))

    # Intepretation Methods

    def weight_plot(self, feature_names):
        ci = np.zeros((2,self.n_features))
        plt.scatter(self.coef_, np.arange(0,self.n_features), c='k', s=20)
        plt.title('Weight plot')
        plt.xlabel('Weight estimate')
        plt.yticks(ticks=np.arange(0,self.n_features),labels=feature_names)
        
        # Compute CI
        alpha = 0.05
        df = self.n_features
        t = stats.t.ppf(1 - alpha/2, df)
        s = np.std(self.coef_, ddof=1)
        n = self.coef_.shape[0]
        ci[0,:] = np.mean(self.theta) - (t * self.sem(self.theta))
        ci[1,:] = np.mean(self.theta) + (t * self.sem(self.theta))

        # ms = marker size, quando è grande il pallino
        # ecolor = il colore della barra del conf int
        # elinewidth = quanto è thicc la barra del conf int
        # capsize = quanto grandi le barre laterali che chiudono il conf int
        # fmt = 'o' significa disegna solo la pallina
        # xerr = tupla con il lower e upper, specificando solo xerr, il conf int viene orizzontale
        plt.errorbar(self.coef_, np.arange(self.n_features), xerr=ci, fmt='o', ms=0, c='k', ecolor='k', elinewidth=1, capsize=2)
        plt.axvline(0, linestyle=':', c='k')
        plt.show()

if __name__ == "__main__":
    # dataset = np.genfromtxt('./Bike-Sharing-Dataset/day.csv',delimiter=',',dtype=None, encoding=None, names=True)
    df = pd.read_csv('./Bike-Sharing-Dataset/day.csv', header=0, parse_dates=True)
    # No uso categorical, lasciamo interi come sono
    # qui fare preprocessing come su github
    df = df.drop(labels=['instant','dteday','registered','casual','atemp'],axis=1)
    print(df)
    exit()
    dataset = load_boston()
    X = dataset.data
    X = StandardScaler().fit(X)
    y = np.reshape(dataset.target, (-1,1))
    labels = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train, epochs=10000, step=0.001)

    # one = np.reshape(X_test[0], (1,-1))

    y_pred = model.predict(X_test)
    
    model.weight_plot(labels)


    
