import numpy as np
import scipy
from scipy import stats
from scipy.special import expit # stable logistic function
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from cost_functions import MeanSquaredErrorLoss, Binary_CrossEntropyLoss
import statistics
import model_interpretable_methods
import matplotlib.pyplot as plt

def back_solver(A,b):
    n = A.shape[1]
    xcomp = np.zeros(n)

    x_n = b[-1] / A[-1,-1]

    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(n-1, i, -1):
            tmp -= xcomp[j]*A[i,j]
            
        xcomp[i] = tmp/A[i,i]
    return xcomp

class LinearModel(object):
    def __init__(self):
        self.error = None
        self.inputs = None
        self.theta = None
        self.intercept_ = None
        self.coef_ = None
        self.n_samples = None
        self.n_features = None

    def standard_error(self): # standard error of the coefficents
        n = self.n_samples
        s = np.sqrt(np.std(self.error) * (1/(n - 2)))
        # se of the bias and coefficients
        se_bias = s/np.sqrt(n) * np.sqrt(1 + (np.mean(self.inputs)**2 / np.var(self.inputs)))
        se_coef = s/np.sqrt(n) * 1/np.std(self.inputs)
        print(self.sse)
        return np.sqrt(self.sse/np.sum((self.inputs - np.mean(self.inputs,axis=0))**2,axis=0))
        
    
    def t_value_abs(self):
        return [np.abs(self.theta[i] / self.se()[i]) for i in range(self.n_features)]
        # return np.abs(self.theta / self.se())


class LinearRegressor(LinearModel):
    def __init__(self,labels):
        super().__init__()
        # self.weights è matrice Nxd+1 (compreso il bias)
        self.labels = labels

    def fit(self, X, y, epochs=1000, step=0.1):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # _X = np.hstack((np.ones((X.shape[0], 1)), X))

        # gradient descent approach
        self.theta = np.random.normal(size=(self.n_features + 1, 1))
        _X = np.hstack((np.ones((X.shape[0], 1)), X))

        for epoch in range(epochs):
           y_pred = _X @ self.theta
           # compute error with mse
           self.error = MeanSquaredErrorLoss().forward(y,y_pred)
           # compute gradient of cost
           # grad = (2/self.n_samples) * (_X.T.dot(_X.dot(self.theta) - y))
           #grad_intercept = (-2/self.n_samples) * np.sum(y_pred-y)
           # penso che il fatto che sulle formule online usino sum dipenda da come implementano theta
           # siccome il mio theta è una matrice, mi serve un vettore di aggiornamento e non uno scalare
           grad = (1/self.n_samples) * np.dot(_X.T,(y_pred-y))
           # update weights
           #self.theta[0] += step * grad_intercept
           self.theta -= step * grad
           print(f'epoch: {epoch+1}, error: {self.error}')

        # Moore-Penrose pseudo-inverse approach
        #self.inputs = np.hstack((np.ones((X.shape[0], 1)),X))
        # compute in a single step, the best parameters that fit the data
        #theta = np.linalg.inv(np.dot(self.inputs.T,self.inputs)).dot(self.inputs.T).dot(y)
        #theta = np.reshape(theta, (-1,1)) # reshape into a 2d vector

        # QR decomposition approach
        #_X = np.hstack((np.ones((X.shape[0], 1)), X))
        # il qr di scipy non ritorna una matrice R quadrata
        #Q, R = np.linalg.qr(self.inputs)
        # theta = scipy.linalg.solve(R,np.dot(Q.T,y))
        #theta = np.dot(np.linalg.inv(R),np.dot(Q.T,y))
        # theta = np.dot(R,np.dot(Q.T,y))
        # theta = scipy.linalg.solve_triangular(R,np.dot(Q.T,y))
        #print('qr theta',theta)
        
        # SVD approach
        # TODO

        # numpy approach
        #self.coef_,residues,_,_ = scipy.linalg.lstsq(X,y)
        #print('lstsq theta',self.coef_)
        
        self.coef_ = self.theta
        self.intercept_ = self.coef_[0]
        # print('theta best', self.theta.shape) (14,1)
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = _X @ self.coef_
        # compute error with mse
        #self.error = MeanSquaredErrorLoss().forward(y,y_pred)
        #print('error of np theta',self.error)
        self.sse = statistics.residual_sum_squares(y,y_pred)
        # residual standard error
        k = self.n_features-1
        self.residual_standard_error = statistics.residual_standard_error(y,y_pred,k)
        self.standard_error = statistics.standard_error(y,y_pred,k,self.coef_)
        
        t_value_abs = np.abs(statistics.t_value(y,y_pred,k,self.coef_))
        self.t_value = t_value_abs

        return self

    def predict(self, X):
        # if passed only a sample instead of test matrix
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return _X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        return statistics.multiple_r_squared(y,y_pred)
        # adjusted r squared
        #return 1 - (1 - r2_biased) * ((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))

    def summary(self):
        l = np.reshape(np.array(['(Intercept)']+self.labels),(-1,1))
        print(stats.t.ppf(self.t_value,self.n_features-1))
        data = np.hstack((l,self.coef_, self.standard_error, self.t_value))
        df = pd.DataFrame(data, columns=['feature','weight','SE','|t|'])
        print(df)



class LogisticBinaryClassifier(LinearModel):
    def __init__(self,labels):
        super().__init__()
        self.labels = labels

    def predict(self, X):
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return expit(_X @ self.coef_)

    def fit(self, X, y, epochs=200, step=0.01):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        theta = np.random.normal(size=(self.n_features + 1, 1))
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        for epoch in range(epochs):
            np.random.shuffle(_X)
            y_pred = expit(_X @ theta)
            #error = - np.sum(np.sum(y * np.log(y_pred),axis=1),axis=0) # multiclass cross-entropy error
            #error = 1/self.n_samples * - np.dot(y.T,np.log(y_pred)) - np.dot((1-y).T,np.log(1-y_pred)) # cross entropy error for binary-class
            error = Binary_CrossEntropyLoss().forward(y_pred,y)
            # qua non ci va la sommatoria proprio da formula
            grad = np.dot(_X.T,(y_pred - y))
            theta -= step * grad
            #print(f'epoch: {epoch+1}, error: {error}')
        self.coef_ = theta
        self.intercept_ = self.coef_[0]

        self.standard_error = statistics.standard_error(y,y_pred,self.n_features,self.coef_)

        P_yes = y_pred
        self.odds = P_yes/(1-P_yes)
        self.odds_ratio = np.exp(self.coef_)
        print(self.odds_ratio)

        # odds_yes = P(y=1)
        # odds_no = 1-P(y=1)
        # odds: odds_yes/odds_no
        # odds_ratio: np.exp(self.coef_)
        
        return self

    def score(self,X,y):
        y_pred = self.predict(X)
        print(y_pred)
        # for binary and multi-class classification
        # this is the jaccard index
        return accuracy_score(y,y_pred)

    def summary(self):
        l = np.reshape(np.array(['(Intercept)']+self.labels),(-1,1))
        data = np.hstack((l, self.coef_, self.odds_ratio, self.standard_error))
        df = pd.DataFrame(data=data, columns=['feature','weights','odds ratio','SE'])
        print(df)
    
class LassoRegressor(LinearModel):
    def __init__(self,labels):
        super().__init__()
        # self.weights è matrice Nxd+1 (compreso il bias)
        self.labels = labels

    def fit(self, X, y, epochs=1000, step=0.1):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # _X = np.hstack((np.ones((X.shape[0], 1)), X))

        # gradient descent approach
        self.theta = np.random.normal(size=(self.n_features + 1, 1))
        _X = np.hstack((np.ones((X.shape[0], 1)), X))

        for epoch in range(epochs):
           y_pred = _X @ self.theta
           # compute error with mse
           self.error = MeanSquaredErrorLoss().forward(y,y_pred) + alpha * np.sum(self.theta)
           # compute gradient of cost
           # grad = (2/self.n_samples) * (_X.T.dot(_X.dot(self.theta) - y))
           #grad_intercept = (-2/self.n_samples) * np.sum(y_pred-y)
           # penso che il fatto che sulle formule online usino sum dipenda da come implementano theta
           # siccome il mio theta è una matrice, mi serve un vettore di aggiornamento e non uno scalare
           grad = (1/self.n_samples) * np.dot(_X.T,(y_pred-y))
           # update weights
           #self.theta[0] += step * grad_intercept
           self.theta -= step * grad
           print(f'epoch: {epoch+1}, error: {self.error}')

        # Moore-Penrose pseudo-inverse approach
        #self.inputs = np.hstack((np.ones((X.shape[0], 1)),X))
        # compute in a single step, the best parameters that fit the data
        #theta = np.linalg.inv(np.dot(self.inputs.T,self.inputs)).dot(self.inputs.T).dot(y)
        #theta = np.reshape(theta, (-1,1)) # reshape into a 2d vector

        # QR decomposition approach
        #_X = np.hstack((np.ones((X.shape[0], 1)), X))
        # il qr di scipy non ritorna una matrice R quadrata
        #Q, R = np.linalg.qr(self.inputs)
        # theta = scipy.linalg.solve(R,np.dot(Q.T,y))
        #theta = np.dot(np.linalg.inv(R),np.dot(Q.T,y))
        # theta = np.dot(R,np.dot(Q.T,y))
        # theta = scipy.linalg.solve_triangular(R,np.dot(Q.T,y))
        #print('qr theta',theta)
        
        # SVD approach
        # TODO

        # numpy approach
        #self.coef_,residues,_,_ = scipy.linalg.lstsq(X,y)
        #print('lstsq theta',self.coef_)
        
        self.coef_ = self.theta
        self.intercept_ = self.coef_[0]
        # print('theta best', self.theta.shape) (14,1)
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = _X @ self.coef_
        # compute error with mse
        #self.error = MeanSquaredErrorLoss().forward(y,y_pred)
        #print('error of np theta',self.error)
        self.sse = statistics.residual_sum_squares(y,y_pred)
        # residual standard error
        k = self.n_features-1
        self.residual_standard_error = statistics.residual_standard_error(y,y_pred,k)
        self.standard_error = statistics.standard_error(y,y_pred,k,self.coef_)
        
        t_value_abs = np.abs(statistics.t_value(y,y_pred,k,self.coef_))
        self.t_value = t_value_abs

        return self

    def predict(self, X):
        # if passed only a sample instead of test matrix
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return _X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        return statistics.multiple_r_squared(y,y_pred)
        # adjusted r squared
        #return 1 - (1 - r2_biased) * ((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))

    def summary(self):
        l = np.reshape(np.array(['(Intercept)']+self.labels),(-1,1))
        print(stats.t.ppf(self.t_value,self.n_features-1))
        data = np.hstack((l,self.coef_, self.standard_error, self.t_value))
        df = pd.DataFrame(data, columns=['feature','weight','SE','|t|'])
        print(df)

class GeneralizedLinearModel(LinearModel):
    pass