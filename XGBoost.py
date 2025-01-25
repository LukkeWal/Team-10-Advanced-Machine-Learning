from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error

##### Code examples: Use XGBoost and plot #####

# import matplotlib.pyplot as plt

# feature_matrices = construct_feature_matrix_array(data)
# trainX = feature_matrices[0][:,0:-2]
# trainY = feature_matrices[0][:,-2]
# testX = feature_matrices[2][:,0:-2]
# testY = feature_matrices[2][:,-2]

# y_pred, scores = XgBoost_meter(trainX, testX, trainY, testY)
# print(scores)

# fig = plt.figure(figsize=(6, 6))

# plt.plot(testY,label='true')
# plt.plot(y_pred,label='pred')

# plt.title("XGBoost for one meter")
# plt.legend()
# plt.show()


##### Main functions #####

def XgBoost(data_train, testData,eta,max_depth):

    data_train = pd.concat(data_train, axis=0, ignore_index=True)
    data_test = pd.concat(testData, axis=0, ignore_index=True)

    y_train = data_train["Peak Value"].values
    X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values

    y_test = data_test["Peak Value"].values
    X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values

    # Fit XGBoost
    model = XGBRegressor(learning_rate = eta, max_depth = max_depth)  # specify paramters
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, y_test

def XgBoost_global(data_meters: list[pd.DataFrame],
                   split_horizontal = False, 
                   size = 0.8):

    """
    This function applies XGBoost to the data for all users (globally) in the following way: data frames
    are concatenated vertically, so essentially we mix all 9-tuples together. The format is the one returned by 
    construct_feature_matrix_DataFrame() (so after applying the sliding window, with 9-tuples as rows).

    The return will be a tuple with:
    - an array with the predicted values for the peak value.
    - the true values to be predicted

    --- Other inputs:

    split_horizontal = Boolean. If true, the function splits the data such that only on the first (size * len(data_meters)) **users**
    are used for training (but with their whole time series). The rest of the users are used for predictions

    size = Float. The fraction of users used for training the data
    """

    if split_horizontal == False:
        # Concatenate all in one DataFrame
        data = pd.concat(data_meters, ignore_index=True, axis=0)

        # Extract relevant values as arrays
        y = data["Peak Value"].values
        X = data.drop(columns=["Peak Value", "Peak Position"]).values

        # Fit XGBoost
        model = XGBRegressor() # specify paramters
        model.fit(X, y)
        y_pred = model.predict(X)

        return y_pred, y
    
    else:
        # Keep the first users only
        data_train = pd.concat(data_meters[:int(size * len(data_meters))], axis=0, ignore_index=True)
        data_test = pd.concat(data_meters[int(size * len(data_meters)):], axis=0, ignore_index=True)

        # Extract relevant values as arrays
        y_train = data_train["Peak Value"].values
        X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values

        y_test = data_test["Peak Value"].values
        X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values

        # Fit XGBoost
        model = XGBRegressor() # specify paramters
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred, y_test


