import numpy as np
from datetime import datetime
from scipy import stats
from scipy.special import expit # stable logistic function
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

        # Intepretation Methods

    def weight_plot(self, feature_names):
        ci = np.zeros((2,self.n_features))
        #plt.scatter(self.coef_, np.arange(0,self.n_features), c='k', s=20)
        plt.title('Weight plot')
        plt.xlabel('Weight estimate')
        plt.yticks(ticks=np.arange(0,self.n_features),labels=feature_names)
        
        # Compute CI
        alpha = 0.05
        df = self.n_features
        t = stats.t.ppf(1 - alpha/2, df)
        s = np.std(self.coef_, ddof=1)
        n = self.coef_.shape[0]
        #ci[0,:] = self.coef_.flatten() - (t * s / np.sqrt(n))
        #ci[1,:] = self.coef_.flatten() + (t * s / np.sqrt(n))
        lower, upper = stats.norm.interval(alpha,loc=self.coef_, scale=s/np.sqrt(n))
        ci[0,:] = lower.flatten()
        ci[1,:] = upper.flatten()
        # ms = marker size, quando è grande il pallino
        # ecolor = il colore della barra del conf int
        # elinewidth = quanto è thicc la barra del conf int
        # capsize = quanto grandi le barre laterali che chiudono il conf int
        # fmt = 'o' significa disegna solo la pallina
        # xerr = tupla con il lower e upper, specificando solo xerr, il conf int viene orizzontale
        plt.errorbar(self.coef_, np.arange(self.n_features), xerr=ci, fmt='o', ms=5, c='k', ecolor='k', elinewidth=1.5, capsize=2.5)
        plt.axvline(0, linestyle=':', c='k')
        plt.show()

    def effect_plot(self, feature_names):
        # calculate the effect for the data
        # self.effects = np.zeros_like(self.inputs)
        self.effects = np.multiply(self.inputs, self.coef_.T)
        #for i,input in enumerate(self.inputs):
        #    self.effects[i,:] = np.multiply(input, self.coef_.T)
        plt.title('Effect plot')
        plt.xlabel('Feature effect')
        plt.yticks(ticks=np.arange(0,self.n_features),labels=feature_names)
        plt.axvline(0, linestyle=':', c='k')
        plt.boxplot(self.effects, vert=False, labels=feature_names)
        plt.show()

class LinearRegressor(LinearModel):
    def __init__(self):
        super().__init__()
        # self.weights è matrice Nxd+1 (compreso il bias)

    def fit(self, X, y, epochs=200, step=0.01):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # _X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.inputs = X

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
        self.weights = np.hstack((np.ones((X.shape[0], 1)),X))
        # compute in a single step, the best parameters that fit the data
        self.theta = np.linalg.inv(np.dot(self.weights.T,self.weights)).dot(self.weights.T).dot(y)
        self.theta = np.reshape(self.theta, (-1,1)) # reshape into a 2d vector

        # QR decomposition approach
        # _X = np.hstack((np.ones((X.shape[0], 1)), X))
        # Q, R = np.linalg.qr(_X)
        # self.theta = np.linalg.inv(R).dot(Q.T).dot(y)

        # SVD approach
        # TODO

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        # print('theta best', self.theta.shape) (14,1)
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = _X @ self.theta
        # compute error with mse
        self.error = (1/self.n_samples) * np.sum((y_pred - y)**2)
        self.sse = np.sum((y - y_pred)**2)
        k = self.n_features-1
        self.residual_standard_error = np.sqrt(self.sse/self.n_samples - (1 + k))
        self.standard_error = self.residual_standard_error / np.sqrt(self.theta)
        print(self.standard_error)
        #self.sst = np.sum((y - np.mean(y))**2)
        s = np.sqrt(self.sse/(self.n_samples - 2))
        #self.se_est = np.sqrt(s) / np.sqrt(np.diag(np.linalg.inv(np.dot(self.weights.T,self.weights))))
        
        #self.se_est = np.reshape(self.se_est, (-1, 1))
        t_value_abs = []
        for i,weight in enumerate(self.theta):
            t_value_abs.append(np.abs(weight/self.standard_error))
        #self.t_value = np.abs(np.divide(self.theta, self.se_est))
        self.t_value = np.array(t_value_abs)
        

        

        return self

    def predict(self, X):
        # if passed only a sample instead of test matrix
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        return _X @ self.theta

    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred)**2)
        sst = np.sum((y - np.mean(y))**2)
        r2_biased = 1 - sse / sst
        return r2_biased
        # adjusted r squared
        #return 1 - (1 - r2_biased) * ((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))

    def summary(self):
        data = np.hstack((self.theta, self.standard_error, self.t_value))
        df = pd.DataFrame(data, columns=['weight','SE','|t|'])
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
    print(model.score(X_test, y_test))

    myModel = LinearRegressor().fit(X_train,y_train)
    print(myModel.score(X_test,y_test))
    myModel.summary()
    # print(model.coef_.T - myModel.coef_)
    # print(myModel.coef_)
    #model.fit(X_train, y_train, epochs=10, step=0.02)
    #myModel.summary(labels)
    # one = np.reshape(X_test[0], (1,-1))

    #y_pred = model.predict(X_test)
    

    # Logistic Regression test

    