import numpy as np
from datetime import datetime
import scipy
from scipy import stats
from scipy.special import expit # stable logistic function
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from cost_functions import MeanSquaredErrorLoss
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
        self.weights = None
        self.error = None
        self.inputs = None
        self.theta = None
        self.intercept_ = None
        self.coef_ = None
        self.n_samples = None
        self.n_features = None
        self.sse = None
        self.sst = None

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

    def fit(self, X, y, epochs=200, step=0.01):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # _X = np.hstack((np.ones((X.shape[0], 1)), X))

        # gradient descent approach
        # self.theta = np.random.normal(size=(self.n_features + 1, 1))
        # _X = np.hstack((np.ones((X.shape[0], 1)), X))

        #for epoch in range(epochs):
        #    y_pred = _X @ self.theta
        #    # compute error with mse
        #    self.error = (1/self.n_samples) * np.sum((y_pred - y)**2)
        #    # compute gradient of cost
        #    grad = (2/self.n_samples) * (_X.T.dot(_X.dot(self.theta) - y))
        #    # update weights
        #    self.theta -= step * grad
        #    print('epoch: {}, error: {}'.format(epoch+1, self.error))

        # Moore-Penrose pseudo-inverse approach
        self.inputs = np.hstack((np.ones((X.shape[0], 1)),X))
        # compute in a single step, the best parameters that fit the data
        #theta = np.linalg.inv(np.dot(self.inputs.T,self.inputs)).dot(self.inputs.T).dot(y)
        #theta = np.reshape(theta, (-1,1)) # reshape into a 2d vector

        # QR decomposition approach
        #_X = np.hstack((np.ones((X.shape[0], 1)), X))
        # il qr di scipy non ritorna una matrice R quadrata
        Q, R = np.linalg.qr(self.inputs)
        # theta = scipy.linalg.solve(R,np.dot(Q.T,y))
        theta = np.dot(np.linalg.inv(R),np.dot(Q.T,y))
        # theta = scipy.linalg.solve_triangular(R,np.dot(Q.T,y))
        print('qr theta',theta)
        
        # SVD approach
        # TODO

        # numpy approach
        #self.coef_,residues,_,_ = scipy.linalg.lstsq(X,y)
        #print('lstsq theta',self.coef_)
        
        self.coef_ = theta
        self.intercept_ = self.coef_[0]
        # print('theta best', self.theta.shape) (14,1)
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = _X @ self.coef_
        # compute error with mse
        self.error = MeanSquaredErrorLoss().forward(y,y_pred)
        print('error of np theta',self.error)
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



class LogisticRegressor(LinearModel):
    def __init__(self):
        super().__init__()

    def predict(self, X):
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return expit(_X @ self.theta)

    def fit(self, X, y, epochs=200, step=0.01):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.inputs = X
        self.theta = np.random.normal(loc=0.1, scale=2, size=(self.n_features + 1,1))
        self.theta[0,:] = 1 # initialize bias (w_0)
        self.error = 0
        for epoch in range(epochs):
            y_pred = self.predict(self.inputs)
            print(y_pred)
            error = - np.sum(np.sum(y * np.log(y_pred),axis=1),axis=0) # multiclass cross-entropy error
            # error = 1/self.n_samples * - np.sum(y * np.log(y_pred)) # cross entropy error for multi-class > 2
            self.error = error
            grad = np.sum((y_pred - y)* self.theta) 
            self.theta -= step * grad
            print('epoch: {}, error: {}'.format(epoch+1, error))
        
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
class LassoRegressor(LinearModel):
    pass

if __name__ == "__main__":
    np.random.seed(1)
    # dataset = np.genfromtxt('./Bike-Sharing-Dataset/day.csv',delimiter=',',dtype=None, encoding=None, names=True)
    df = pd.read_csv('./Bike-Sharing-Dataset/day.csv', header=0, parse_dates=True)
    
    # qui fare preprocessing come su github
    df['season'] = df['season'].map({2:'seasonSUMMER', 3:'seasonFALL',4:'seasonWINTER'})
    df['holiday'] = df['holiday'].map({1:'holidayHOLIDAY'})
    df['workingday'] = df['workingday'].map({1:'workingdayWORKING DAY'})
    df['weathersit'] = df['weathersit'].map({1:'weathersitGOOD',2:'weathersitMISTY',3:'weathersitRAIN/SNOW/STORM'})
    df = pd.get_dummies(df, prefix='', prefix_sep='',columns=['season','holiday','workingday','weathersit'])
    
    # weekday to categorical
    df.loc[df['weekday'] == 0, ['weekday']] = 'SUN'
    df.loc[df['weekday'] == 1, ['weekday']] = 'MON'
    df.loc[df['weekday'] == 2, ['weekday']] = 'TUE'
    df.loc[df['weekday'] == 3, ['weekday']] = 'WED'
    df.loc[df['weekday'] == 4, ['weekday']] = 'THU'
    df.loc[df['weekday'] == 5, ['weekday']] = 'FRI'
    df.loc[df['weekday'] == 6, ['weekday']] = 'SAT'
    
    # mnth to categorical
    df.loc[df['mnth'] == 1, ['mnth']] = 'JAN'
    df.loc[df['mnth'] == 2, ['mnth']] = 'FEB'
    df.loc[df['mnth'] == 3, ['mnth']] = 'MAR'
    df.loc[df['mnth'] == 4, ['mnth']] = 'APR'
    df.loc[df['mnth'] == 5, ['mnth']] = 'MAY'
    df.loc[df['mnth'] == 6, ['mnth']] = 'JUN'
    df.loc[df['mnth'] == 7, ['mnth']] = 'JUL'
    df.loc[df['mnth'] == 8, ['mnth']] = 'AUG'
    df.loc[df['mnth'] == 9, ['mnth']] = 'SEP'
    df.loc[df['mnth'] == 10, ['mnth']] = 'OKT'
    df.loc[df['mnth'] == 11, ['mnth']] = 'NOV'
    df.loc[df['mnth'] == 12, ['mnth']] = 'DEZ'
    df['mnth'] = df['mnth'].astype('category')
    # yr to categorical
    df.loc[df['yr'] == 0, ['yr']] = '2011'
    df.loc[df['yr'] == 1, ['yr']] = '2012'
    
    df['days_since_2011'] = [(datetime.strptime(dte,'%Y-%m-%d') - datetime.strptime(min(df['dteday']),'%Y-%m-%d')).days for dte in df['dteday']]
    df['temp'] = df['temp'] * (39 - (- 8)) + (- 8)
    df['atemp'] = df['atemp'] * (50 - (16)) + (16)
    df['windspeed'] = 67 * df['windspeed']
    df['hum'] = 100 * df['hum']
    
    X = df.drop(labels=['instant','dteday','registered','casual','atemp' ,'mnth','yr','weekday'],axis=1)
    X = X.drop(labels=['cnt'],axis=1)
    
    labels = list(X.columns)
    X = X.to_numpy()
    X = StandardScaler().fit_transform(X)
    y = np.reshape(df['cnt'].to_numpy(), (-1,1))

    #dataset = load_boston()
    #print(dataset.DESCR)
    #X = dataset.data
    #X = StandardScaler().fit_transform(X)
    #y = np.reshape(dataset.target, (-1,1))
    #labels = dataset.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)

    model = LinearRegression().fit(X_train,y_train)
    print('model score',model.score(X_test, y_test))

    myModel = LinearRegressor(labels=labels).fit(X_train,y_train)
    print('mymodel score',myModel.score(X_test,y_test))
    myModel.summary()
    model_interpretable_methods.weight_plot(myModel.coef_,y,myModel.y_pred,myModel.n_features,labels)
    # print(model.coef_.T - myModel.coef_)
    # print(myModel.coef_)
    #model.fit(X_train, y_train, epochs=10, step=0.02)
    #myModel.summary(labels)
    # one = np.reshape(X_test[0], (1,-1))

    #y_pred = model.predict(X_test)
    

    # Logistic Regression test

    