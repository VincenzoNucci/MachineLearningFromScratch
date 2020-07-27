import numpy as np
from datetime import datetime
from scipy import stats
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from preprocessing_functions import StandardScaler

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

    def sees(self): # standard error of the coefficents
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
    def __init__(self):
        super().__init__()
        # self.weights è matrice Nxd+1 (compreso il bias)

    def fit(self, X, y, epochs=200, step=0.01):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.inputs = X
        # define matrix w with instances plus bias
        # self.theta = np.hstack((np.ones((X.shape[0], 1)),X))
        self.theta = np.random.normal(size=(self.n_features + 1, 1))
        _X = np.hstack((np.ones((X.shape[0], 1)), X))

        for epoch in range(epochs):
            y_pred = _X @ self.theta
            # compute error with mse
            self.error = (1/self.n_samples) * np.sum((y_pred - y)**2)
            # compute gradient of cost
            grad = (2/self.n_samples) * (_X.T.dot(_X.dot(self.theta) - y))
            # update weights
            self.theta -= step * grad
            print('epoch: {}, error: {}'.format(epoch+1, self.error))
            
            #errore = self.cost.forward(self.predict(X),y)
            #print('epoch: {}, error: {}'.format(epoch+1, errore))
            #error.append(errore)
            #self.theta -= step * self.cost.backward(errore, y)
        
        self.weights = np.hstack((np.ones((X.shape[0], 1)),X))
        # compute in a single step, the best parameters that fit the data
        self.theta = np.linalg.inv(np.dot(self.weights.T,self.weights)).dot(self.weights.T).dot(y)
        self.theta = np.reshape(self.theta, (-1,1)) # reshape into a 2d vector
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        # print('theta best', self.theta.shape) (14,1)
        y_pred = _X @ self.theta
        # compute error with mse
        self.error = (1/self.n_samples) * np.sum((y_pred - y)**2)
        self.sse = np.sum((y - y_pred)**2)
        self.sst = np.sum((y - np.mean(y)**2))
        s = np.sqrt(self.sse/(self.n_samples - 2))
        self.se_est = np.sqrt(1 / (self.n_samples-2) * np.sum((y_pred - y)**2,axis=0) / np.sum((self.inputs - np.mean(self.inputs))**2,axis=0))
        self.se_est = np.reshape(self.se_est, (-1, 1))
        print(self.se_est)
        self.t_value = np.abs(np.divide(self.theta, self.se_est))
        print(np.hstack((self.theta, self.se_est, self.t_value)))

        

        return self

    def predict(self, X):
        # if passed only a sample instead of test matrix
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return _X @ self.theta

    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred)**2)
        sst = np.sum((y - np.mean(y)**2))
        r2_biased = 1 - sse / sst
        return 1 - (1 - r2_biased) * ((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))

    def summary(self):
        summary = np.zeros((self.n_features + 1, 4))
        print(self.theta)
        print(self.se_est)
        print(self.t_value)
        

    # Intepretation Methods

    def weight_plot(self, coefs_, n_features, feature_names):
        ci = np.zeros((2,self.n_features))
        plt.scatter(self.coef_, np.arange(0,self.n_features), c='k', s=20)
        plt.title('Weight plot')
        plt.xlabel('Weight estimate')
        plt.yticks(ticks=np.arange(0,self.n_features),labels=feature_names)
        
        # Compute CI
        alpha = 0.05
        df = self.n_features
        t = stats.t.ppf(1 - alpha/2, df)
        s = np.std(self.theta, ddof=1)
        n = self.theta.shape[0]
        ci[0,:] = np.mean(self.theta) - (t * s / np.sqrt(n))
        ci[1,:] = np.mean(self.theta) + (t * s / np.sqrt(n))

        # ms = marker size, quando è grande il pallino
        # ecolor = il colore della barra del conf int
        # elinewidth = quanto è thicc la barra del conf int
        # capsize = quanto grandi le barre laterali che chiudono il conf int
        # fmt = 'o' significa disegna solo la pallina
        # xerr = tupla con il lower e upper, specificando solo xerr, il conf int viene orizzontale
        plt.errorbar(self.coef_, np.arange(self.n_features), xerr=ci, fmt='o', ms=0, c='k', ecolor='k', elinewidth=1.5, capsize=2.5)
        plt.axvline(0, linestyle=':', c='k')
        plt.show()

if __name__ == "__main__":
    # dataset = np.genfromtxt('./Bike-Sharing-Dataset/day.csv',delimiter=',',dtype=None, encoding=None, names=True)
    df = pd.read_csv('./Bike-Sharing-Dataset/day.csv', header=0, parse_dates=True)
    # No uso categorical, lasciamo interi come sono
    # qui fare preprocessing come su github
    # season to categorical
    df['seasonSUMMER'] = 0
    df['seasonFALL'] = 0
    df['seasonWINTER'] = 0

    df.loc[df['season'] == 2, ['seasonSUMMER']] = 1
    df.loc[df['season'] == 3, ['seasonFALL']] = 1
    df.loc[df['season'] == 4, ['seasonWINTER']] = 1
    # weekday to categorical
    df.loc[df['weekday'] == 0, ['weekday']] = 'SUN'
    df.loc[df['weekday'] == 1, ['weekday']] = 'MON'
    df.loc[df['weekday'] == 2, ['weekday']] = 'TUE'
    df.loc[df['weekday'] == 3, ['weekday']] = 'WED'
    df.loc[df['weekday'] == 4, ['weekday']] = 'THU'
    df.loc[df['weekday'] == 5, ['weekday']] = 'FRI'
    df.loc[df['weekday'] == 6, ['weekday']] = 'SAT'
    # holiday to categorical
    df['holidayHOLIDAY'] = df['holiday']
    # df.loc[df['holiday'] == 0, ['holiday']] = 'NO HOLIDAY'
    # df.loc[df['holiday'] == 1, ['holiday']] = 'HOLIDAY'
    # workingday to categorical
    df['workingdayWORKING DAY'] = df['workingday']
    # df.loc[df['workingday'] == 0, ['workingday']] = 'NO WORKING DAY'
    # df.loc[df['workingday'] == 1, ['workingdayWORKING DAY']] = 1
    # weathersit to categorical
    df['weathersitGOOD'] = 0
    df['weathersitMISTY'] = 0
    df['weathersitRAIN/SNOW/STORM'] = 0
    df.loc[df['weathersit'] == 1, ['weathersitGOOD']] = 1
    df.loc[df['weathersit'] == 2, ['weathersitMISTY']] = 1
    df.loc[df['weathersit'] == 3, ['weathersitRAIN/SNOW/STORM']] = 1
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
    
    X = df.drop(labels=['instant','dteday','registered','casual','atemp',  'holiday','workingday','weathersit','mnth','yr','season','weekday'],axis=1)
    X = X.drop(labels=['cnt'],axis=1)
    
    labels = list(X.columns)
    X = X.to_numpy()
    X = StandardScaler().fit(X)
    y = np.reshape(df['cnt'].to_numpy(), (-1,1))

    #dataset = load_boston()
    #X = dataset.data
    #X = StandardScaler().fit(X)
    #y = np.reshape(dataset.target, (-1,1))
    #labels = dataset.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)

    myModel = LinearRegressor().fit(X_train, y_train, epochs=1)
    model = LinearRegression().fit(X_train,y_train)
    #print(model.score(X_test, y_test))
    print(myModel.score(X_test, y_test))
    #print(model.coef_.T - myModel.coef_)
    # print(myModel.coef_)
    #model.fit(X_train, y_train, epochs=10, step=0.02)
    myModel.summary()
    # one = np.reshape(X_test[0], (1,-1))

    #y_pred = model.predict(X_test)
    
    #model.weight_plot(labels)


    
