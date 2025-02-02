import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error, accuracy_score
from PrepareData import read_data
from LinearRegression import linear_regression
from XGBoost import XgBoost_regression, XgBoost_classifier, xgBoost_cv_hyp_search
from sklearn.preprocessing import OneHotEncoder


from pdb import set_trace

# our best parameters
best_regression_params = {
    'max_depth': [7],
    'min_child_weight': [6.48687005],
    'learning_rate': [0.09914108],
    'subsample': [0.50536943],
    'colsample_bytree': [0.70956684],
    'reg_lambda': [2.52174128],
    'gamma': [1.51326692]
    }
    
best_classification_params = {
    'max_depth': [7],
    'min_child_weight': [6.35119954],
    'learning_rate': [0.05147735],
    'subsample': [0.58568049],
    'colsample_bytree': [0.89649074],
    'reg_lambda': [2.30147386],
    'gamma': [0.73406308]
    }

def main():
    
    ##### LOAD THE DATA #####

    start_time = time.time()
    data = read_data("ProcessedData")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"processed data in {elapsed_time:.4f} seconds")


    size = 0.8
    data_test = data[int(size * len(data)):]  # for final testing
    data_train = data[:int(size * len(data))]  # for cross validation
    print(xgBoost_cv_hyp_search('regression'),data_train)
    print(xgBoost_cv_hyp_search('classification'),data_train)



if __name__ == "__main__":
    main()
