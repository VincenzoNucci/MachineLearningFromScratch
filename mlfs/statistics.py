import numpy as np

def residual_sum_squares(y,y_pred):
    '''
    sse
    '''
    return np.sum((y - y_pred)**2)

def total_sum_squares(y):
    '''
    sst
    '''
    return np.sum((y - np.mean(y))**2)

def residual_standard_error(y,y_pred,k):
    '''
    n number of samples
    k number of features minus the intercept
    sse the residual sum of squares
    '''
    n = y.shape[0]
    return np.sqrt(residual_sum_squares(y,y_pred)/(n - (1 + k)))

def standard_error(y,y_pred,k,x):
    '''
    Std. Error is Residual Standard Error divided by the square root of the sum of the square of that particular x variable.
    '''
    return residual_standard_error(y,y_pred,k) / np.sqrt((x**2))

def t_value(y,y_pred,k,x):
    return x / standard_error(y,y_pred,k,x)

def multiple_r_squared(y,y_pred):
    '''
    coefficient of determination r^2 of the prediction

    problem: the more the features, the higher the score, solution use adjusted_r_squared
    '''
    return 1 - (residual_sum_squares(y,y_pred) / total_sum_squares(y))

def adjusted_r_squared(y,y_pred,n,k):
    '''
    n number of samples
    k number of features
    '''
    return 1 - (multiple_r_squared(y,y_pred)) * ((n - 1)/(n - (k + 1)))