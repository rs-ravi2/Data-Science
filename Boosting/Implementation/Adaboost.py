import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def calculate_alpha(error): 
    return np.log((1 - error) / error)

def calculate_error(h_x, y, w_i):
    return (sum(w_i * (np.not_equal(y, h_x)).astype(int)))/sum(w_i)


def update_weights(w_i, alpha, y, h_x):
    return w_i * np.exp(alpha * (np.not_equal(y, h_x)).astype(int))


class Adaboost:
    
    def __init__(self):
        # self.w_i = None
        self.alphas = []
        self.H_t = []
        self.t = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, t = 400):
        
        self.alphas = [] 
        self.training_errors = []
        self.t = t

        for t in range(0, t):
            
            if t == 0:
                w_i = np.ones(len(y)) * 1 / len(y) # Initialise weights
            else:
                w_i = update_weights(w_i, alpha_m, y, h_x)
            # print(w_i)
            
            H_t = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            H_t.fit(X, y, sample_weight = w_i)
            h_x = H_t.predict(X)
            
            self.H_t.append(H_t)

            error_m = calculate_error(y, h_x, w_i)
            self.training_errors.append(error_m)
            # print(error_m)

            alpha_m = calculate_alpha(error_m)
            self.alphas.append(alpha_m)
            # print(alpha_m)

        assert len(self.H_t) == len(self.alphas)


    def predict(self, X):

        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.t)) 

        for t in range(self.t):
            h_x_m = self.H_t[t].predict(X) * self.alphas[t]
            weak_preds.iloc[:,t] = h_x_m

        h_x = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return h_x
      
    def error_rates(self, X, y):
        self.prediction_errors = [] # Clear before calling
        
        for t in range(self.t):
            h_x_m = self.H_t[t].predict(X)          
            error_m = calculate_error(y = y, h_x = h_x_m, w_i = np.ones(len(y)))
            self.prediction_errors.append(error_m)