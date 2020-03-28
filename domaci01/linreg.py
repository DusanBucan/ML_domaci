import numpy as np
import pandas as pd
import math

def fit(x, y):
    # TODO: Implement this
    pass

def predict(x):
    # TODO: Implement this
    pass

def calculate_rmse(y_true, y_predict):
    rmse = 0
    N = len(y_true)
    sum = 0
    
    for i in range(N):
        sum += (y_predict[i] - y_true[i])**2
    
    rmse = math.sqrt(sum / N)
    
    return rmse
