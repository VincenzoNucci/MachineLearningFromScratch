from linear_models import LogisticBinaryClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

if __name__ == "__main__":
    np.random.seed(0)
    # dataset = np.genfromtxt('./Bike-Sharing-Dataset/day.csv',delimiter=',',dtype=None, encoding=None, names=True)
    df = pd.read_csv('./Cervical-Cancer-Dataset/risk_factors_cervical_cancer.csv', header=0, parse_dates=True)
    df = df.replace(to_replace='?',value=df.mode(axis=1).iloc[0])
    # imputing with the mode may add another '?' if it is the most frequent value
    df = df.replace(to_replace='?',value=0)
    
    # qui fare preprocessing come su github
    df = df.drop(columns=['Hinselmann','Schiller','Citology'],axis=1)
    # df['Biopsy'] = df['Biopsy'].map({0:'Healty',1:'Cancer'})
    
    X = df[['Hormonal Contraceptives', 'Smokes','Num of pregnancies','STDs (number)','IUD']]
    labels = list(X.columns)
    X = X.to_numpy()
    print(np.where(X == '?'))
    X = StandardScaler().fit_transform(X)
    y = np.reshape(df['Biopsy'].to_numpy(), (-1,1))

    #dataset = load_boston()
    #print(dataset.DESCR)
    #X = dataset.data
    #X = StandardScaler().fit_transform(X)
    #y = np.reshape(dataset.target, (-1,1))
    #labels = dataset.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)

    model = LogisticRegression(penalty='none').fit(X_train,y_train)
    print('model score',model.score(X_test, y_test))

    myModel = LogisticBinaryClassifier(labels=labels).fit(X_train,y_train)
    #print('mymodel score',myModel.score(X_test,y_test))
    myModel.summary()
    #model_interpretable_methods.weight_plot(myModel.coef_,y,myModel.y_pred,myModel.n_features,labels)
    
    # print(model.coef_.T - myModel.coef_)
    # print(myModel.coef_)
    #model.fit(X_train, y_train, epochs=10, step=0.02)
    #myModel.summary(labels)
    # one = np.reshape(X_test[0], (1,-1))

    #y_pred = model.predict(X_test)