import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

def plotting_cue(data, idx_feature, labels):
    uniques = data.iloc[:,idx_feature].unique()
    uniques.sort()
    dic = {}
    for i in uniques:
        scores = labels[data.iloc[:,idx_feature] == i].mean()
        dic['Scary{0}'.format(i)] = scores[1]
        dic['Happy{0}'.format(i)] = scores[2]
        dic['Sad{0}'.format(i)] = scores[3]
        dic['Peaceful{0}'.format(i)] = scores[4]
        
    scary_x = []
    scary_y = []
    happy_x = []
    happy_y = []
    sad_x = []
    sad_y = []
    peaceful_x = []
    peaceful_y = []
    for i in dic.items():
        if i[0][:-1] == "Scary":
            scary_x.append(int(i[0][-1]))
            scary_y.append(i[1])
        elif i[0][:-1] == "Happy":
            happy_x.append(int(i[0][-1]))
            happy_y.append(i[1])
        elif i[0][:-1] == "Sad":
            sad_x.append(int(i[0][-1]))
            sad_y.append(i[1])
        elif i[0][:-1] == "Peaceful":
            peaceful_x.append(int(i[0][-1]))
            peaceful_y.append(i[1])
    
    return [scary_x, happy_x, sad_x, peaceful_x], [scary_y, happy_y, sad_y, peaceful_y]

    plotting_cue(data, 2, labels)

class linear_regression:
    def __init__ (self, multi=False):
        self.multi = multi

    def fit(self, data, y):
        if self.multi:
            n = data.shape[1]
            X = np.zeros((data.shape[0], n + 1))
            X[:,1:] = data
        else:
            n = 1
            X = np.zeros((data.shape[0], n + 1))
            X[:,1] = data
        X[:,0] = 1

        w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, y))
        self.w = w
        self.X = X
        self.y = y

    def predict(self, X):
        if len(X.shape) == 1:
            return np.dot(self.w[1], X) + self.w[0]
        predictions = []
        try: 
            for v in X:
                predictions.append(np.dot(self.w[1:], v) + self.w[0])
        except UnboundLocalError:
            raise ValueError('Fit model before predicting')
        return predictions

    def error(self, X, y):
        c_sum = 0
        try: 
            for idx, el in enumerate(X):
                dif = (np.dot(self.w[1:], el) + self.w[0] - y[idx])
                c_sum += dif ** 2
        except UnboundLocalError:
            raise ValueError('Fit model before error measurement')
        return np.sqrt(c_sum / y.shape[0])
    
    def r2_adjusted(self, X, y):
        n = len(y)
        k = len(self.w) - 1

        self.y_mean = np.mean(y)
        total_sum_squared = sum([(c_y - self.y_mean)**2 for c_y in y])
        predicted = self.predict(X)
        residual_sum = sum([(c_y - p_y)**2 for c_y, p_y in zip(y, predicted)])

        r2 = 1 - (residual_sum/total_sum_squared)

        return 1 - ( ((1-r2)*(n-1))/(n-k-1) )

def semi_partial_correlation_squared(X, Y):
    spc = np.zeros((X.shape[1]))
    for idx, el in enumerate(X.T):
        indel = list(range(len(spc)))
        indel.pop(idx)
        temp_set = X[:,indel]
        top = pearsonr(el, Y)[0]
        bottom = 1
        for v in temp_set.T:
            top -= pearsonr(v, Y)[0] * pearsonr(el, v)[0]
            bottom -= pearsonr(el, v)[0]**2
        spc[idx] = (top/np.sqrt(bottom)) ** 2
    return spc

def r2_adjusted(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_sum_squared = sum([(c_y - y_mean)**2 for c_y in y_true])
    residual_sum = sum([(c_y - p_y)**2 for c_y, p_y in zip(y_true, y_pred)])
    return 1 - (residual_sum/total_sum_squared)

def pca(data):
    #subtract mean    
    feature_means = [[] for a in range(data.shape[1])]
    for i, _ in enumerate(feature_means):
        feature_means[i] = data[:,i].mean()
        
    centred = np.empty(data.shape)
    
    for i, el in np.ndenumerate(data):
        centred[i] = el - feature_means[i[1]]    

    #calculate covariance matrix
    comat = np.cov(centred, rowvar=False)#, bias=True)
    w, v = np.linalg.eig(comat)

    idx = w.argsort()[::-1]
    
    eigenvalues = w[idx]
    eigenvectors = v[:,idx]
    return eigenvalues, eigenvectors.T, centred, feature_means
#%%