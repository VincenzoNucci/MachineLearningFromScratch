import numpy as np
import pandas as pd
import json
from copy import copy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from statistics import standard_error
import scipy.stats as stats
with open('cards.json','r') as g:
    cards = json.load(g)
with open('relics.json','r') as g:
    relics = json.load(g)
characters_dict = {
    'IRONCLAD':1,
    'THE_SILENT':2,
    'DEFECT':3,
    'WATCHER':4
}
# cards_columns = [str(card['Name']).replace(' ','') for card in cards]
# relics_columns = [str(relic['Name']).replace(' ','') for relic in relics]
cards_columns = list(cards.keys())
relics_columns = list(relics.keys()) + ['Wing Boots']

for card_name in copy(cards_columns):
    cards_columns.append(card_name+'+1')
columns = [
    'gold_spent',
    'floor_reached',
    'items_purged_num',
    'campfires_num',
    'total_cards',
    'total_relics',
    'potions_used_num',
    'total_damage_taken',
    'character_chosen',
    'items_purchased_num',
    'campfire_rested_num',
    'campfire_upgraded_num',
    'victory'] + cards_columns + relics_columns
from keras import layers
from keras import models
from keras import optimizers

def weight_plot(coef_,y,y_pred,labels):
    k = len(labels)
    ci = np.zeros((2,k+1))
    #plt.scatter(self.coef_, np.arange(0,self.n_features), c='k', s=20)
    plt.title('Weight plot')
    plt.xlabel('Weight estimate')
    plt.yticks(ticks=np.arange(0,k),labels=labels)
    
    # Compute CI
    alpha = 0.05
    df = k
    #t = stats.norm.ppf(alpha/2, df)
    #s = np.std(coef_, ddof=1)
    #n = coef_.shape[0]
    #ci[0,:] = self.coef_.flatten() - (t * s / np.sqrt(n))
    #ci[1,:] = self.coef_.flatten() + (t * s / np.sqrt(n))
    # lower, upper = stats.norm.interval(alpha,loc=self.coef_, scale=s/np.sqrt(n))
    ci[0,:] = (coef_ - stats.norm.ppf(alpha/2,df) * standard_error(y,y_pred,k,coef_)).flatten()
    ci[1,:] = (coef_ + stats.norm.ppf(alpha/2,df) * standard_error(y,y_pred,k,coef_)).flatten()

    # ms = marker size, quando è grande il pallino
    # ecolor = il colore della barra del conf int
    # elinewidth = quanto è thicc la barra del conf int
    # capsize = quanto grandi le barre laterali che chiudono il conf int
    # fmt = 'o' significa disegna solo la pallina
    # xerr = tupla con il lower e upper. specificando solo xerr, il conf int viene orizzontale
    plt.errorbar(coef_, np.arange(k+1), xerr=ci, fmt='o', ms=5, c='k', ecolor='k', elinewidth=1.5, capsize=2.5)
    plt.axvline(0, linestyle=':', c='k')
    plt.show()

# data = pickle.load(open('hearthstone_data.pickle','rb'))
df = pd.read_csv('sts_data.csv',header=0)

X = df.drop(labels=['victory'],axis=1).to_numpy()
y = df['victory'].to_numpy()

X = StandardScaler().fit_transform(X)
print('X shape',X.shape)
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

rf = RandomForestClassifier(max_depth=5)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=0)
rf = rf.fit(X_train,y_train)
# same as using accuracy_score
print(rf.score(X_test,y_test))
plt.bar(range(len(columns)),rf.feature_importances_)
plt.show()

i = np.random.randint(0,X_test.shape[0])
explainer = LimeTabularExplainer(training_data=X_train,feature_names=columns,class_names=['victory','defeat'])
explanation = explainer.explain_instance(X_test[i], rf.predict_proba, num_features=2)
print(explanation.as_list())

logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
print(logreg.score(X_test,y_test))
print(np.argmax(logreg.coef_))

scores = cross_val_score(rf, X_test, y_test, cv=7)
print('Accuracy',scores.mean())

scores = cross_val_score(logreg,X_test,y_test,cv=7)
print('Accuracy',scores.mean())

# y_pred = logreg.predict(X_test)
# print(logreg.get_params().keys())
# train_scores, valid_scores = validation_curve(logreg,X_train, y_train, 'C', np.linspace(0,1) ,cv=5)
# fig, ax = plt.subplots()
# ax.plot(train_scores,label='Train score'),
# ax.plot(valid_scores,label='Valid score')
# plt.show()
# weight_plot(logreg.coef_,y_test,y_pred,columns)

# xgd = GradientBoostingClassifier()
# xgd = xgd.fit(X,y)
# print(xgd.score(X_test,y_test))

dt = DecisionTreeClassifier(max_depth=3)
dt = dt.fit(X_train, y_train)
print(dt.score(X_test,y_test))
plot_tree(dt,feature_names=columns,class_names=['victory','defeat'])
plt.bar(range(len(columns)),dt.feature_importances_)
plt.show()

model = models.Sequential()
model.add(layers.Dense(512, input_shape=(1055,), activation='relu', name='fc1'))
model.add(layers.Dense(1024, activation='relu', name='fc2'))
model.add(layers.Dense(32, activation='relu', name='fc3'))
model.add(layers.Dense(1, activation='softmax', name='output'))

optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=200)

# Test on unseen data

results = model.evaluate(X_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))