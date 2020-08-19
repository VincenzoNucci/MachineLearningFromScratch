from linear_models import LinearRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

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

    #myModel = LinearRegressor(labels=labels).fit(X_train,y_train)
    #print('mymodel score',myModel.score(X_test,y_test))
    #myModel.summary()
    #model_interpretable_methods.weight_plot(myModel.coef_,y,myModel.y_pred,myModel.n_features,labels)
    # print(model.coef_.T - myModel.coef_)
    # print(myModel.coef_)
    #model.fit(X_train, y_train, epochs=10, step=0.02)
    #myModel.summary(labels)
    # one = np.reshape(X_test[0], (1,-1))

    #y_pred = model.predict(X_test)