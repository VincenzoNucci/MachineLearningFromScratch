import numpy as np
from statistics import standard_error
import matplotlib.pyplot as plt
from scipy import stats

def effect_plot():
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

def weight_plot(coef_y,y_pred,k,labels):
    ci = np.zeros((2,k))
    #plt.scatter(self.coef_, np.arange(0,self.n_features), c='k', s=20)
    plt.title('Weight plot')
    plt.xlabel('Weight estimate')
    plt.yticks(ticks=np.arange(0,k),labels=labels)
    
    # Compute CI
    alpha = 0.05
    df = k - 1
    t = stats.norm.ppf(alpha/2, df)
    s = np.std(coef_, ddof=1)
    n = coef_.shape[0]
    #ci[0,:] = self.coef_.flatten() - (t * s / np.sqrt(n))
    #ci[1,:] = self.coef_.flatten() + (t * s / np.sqrt(n))
    # lower, upper = stats.norm.interval(alpha,loc=self.coef_, scale=s/np.sqrt(n))
    ci[0,:] = coef_ - stats.norm.ppf(alpha/2,df) * standard_error(y,y_pred,k,coef_)
    ci[1,:] = coef_ + stats.norm.ppf(alpha/2,df) * standard_error(y,y_pred,k,coef_)
    # ms = marker size, quando è grande il pallino
    # ecolor = il colore della barra del conf int
    # elinewidth = quanto è thicc la barra del conf int
    # capsize = quanto grandi le barre laterali che chiudono il conf int
    # fmt = 'o' significa disegna solo la pallina
    # xerr = tupla con il lower e upper. specificando solo xerr, il conf int viene orizzontale
    plt.errorbar(coef_, np.arange(k), xerr=ci, fmt='o', ms=5, c='k', ecolor='k', elinewidth=1.5, capsize=2.5)
    plt.axvline(0, linestyle=':', c='k')
    plt.show()