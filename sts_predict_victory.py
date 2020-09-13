import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import optimizers

# data = pickle.load(open('hearthstone_data.pickle','rb'))
df = pd.read_csv('sts_data.csv',header=0)

X = df.drop(labels=['victory'],axis=1).to_numpy()
y = df['victory'].to_numpy()

rf = RandomForestClassifier(max_depth=5)
# X = []
# y = []
# np.random.shuffle(data)
# for Xy in data:
#     X.append(Xy[:,:8])
#     y.append(Xy[:,8])
#     # X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=0)
#     # rf = rf.fit(X,y)
#     # print(rf.score(X_test,y_test))
# X_train = np.array(X)[0:200]
# y_train = np.array(y)[0:200]
# X_test = np.array(X)[200:220]
# y_test = np.array(y)[200:220]
# X_valid = np.array(X)[220:278]
# y_valid = np.array(y)[220:278]

# for x,Y in zip(X_train,y_train):
#     rf = rf.fit(x,Y)
# # X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=0)
# # rf = rf.fit(X_train,y_train)
# print(np.mean([rf.score(Xte,yte) for Xte,yte in zip(X_test,y_test)]))
# for vlad,real in zip(X_valid,y_valid):
#     print(f'y_pred: {rf.predict(vlad)}, y: {y_valid}')

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=0)
rf = rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
print(rf.feature_importances_)

logreg = LogisticRegression()
from sklearn.preprocessing import StandardScaler, scale
scaled_X_train = scale(X_train)
logreg = logreg.fit(scaled_X_train,y_train)
scaled_X_test = scale(X_test)
print(logreg.score(scaled_X_test,y_test))
print(logreg.coef_)

# xgd = GradientBoostingClassifier()
# xgd = xgd.fit(X,y)
# print(xgd.score(X_test,y_test))

# model = models.Sequential()
# model.add(layers.Dense(64, input_shape=(9,), activation='relu', name='fc1'))
# model.add(layers.Dense(32, activation='relu', name='fc2'))
# model.add(layers.Dense(1, activation='softmax', name='output'))

# optimizer = optimizers.Adam(lr=0.001)
# model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# print('Neural Network Model Summary: ')
# print(model.summary())

# # Train the model
# model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=2)

# # Test on unseen data

# results = model.evaluate(X_test, y_test)
# print('Final test set loss: {:4f}'.format(results[0]))
# print('Final test set accuracy: {:4f}'.format(results[1]))