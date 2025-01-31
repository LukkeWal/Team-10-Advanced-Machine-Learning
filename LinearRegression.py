import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from PrepareData import split_train_and_test

##### Code examples: Use linear regression and plot #####

# from sklearn.metrics import root_mean_squared_error
# import matplotlib.pyplot as plt

# # Test for a meter
# data_meter_1 = data_processed[0]
# y_pred, y_true = linregress_meter(data_meter_1, split_vertical=True)

# print(y_true.mean(), y_true.std())
# print(f"RMSE: {root_mean_squared_error(y_pred, y_true)}")

# fig = plt.figure(figsize=(6, 6))

# plt.plot(y_true, color="blue", label="True")
# plt.plot(y_pred, color="red", label="Prediction")

# plt.title("Meter 0 - local prediction")
# plt.legend()
# plt.show()

# # Global prediction
# y_pred, y_true = linregress_global(data_processed, split_horizontal=True)

# print(y_true.mean(), y_true.std())
# print(f"RMSE: {root_mean_squared_error(y_pred, y_true)}")

# fig = plt.figure(figsize=(6, 6))

# plt.plot(y_true, color="blue", label="True")
# plt.plot(y_pred, color="red", label="Prediction")

# plt.title("Last 80% Meters - Global prediction")
# plt.legend()
# plt.show()

##### Main functions #####

def linear_regression(data_train, testData):

    """

    This function applies linear regression to the data for all users (globally) in the following way: data frames
     are concatenated vertically, so essentially we mix all 9-tuples together. The format is the one returned by
     construct_feature_matrix_DataFrame() (so after applying the sliding window, with 9-tuples as rows).

     The return will be a tuple with:
     - an array with the predicted values for the peak value.
     - the true values to be predicted

    """

    data_train = pd.concat(data_train, axis=0, ignore_index=True)
    data_test = pd.concat(testData, axis=0, ignore_index=True)

    y_train = data_train["Peak Value"].values
    X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values

    y_test = data_test["Peak Value"].values
    X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, y_test



# Linear Regression for one user
# def linregress_meter(data_meter: pd.DataFrame,
#                      split_vertical = False,
#                      size = 0.8):
#     """
#     This function applies linear regression to the data for one user/meter, which is in
#     the format returned by construct_feature_matrix_DataFrame() (so after applying the sliding
#     window, with 9-tuples as rows).
#
#     The return will be a tuple with:
#     - an array with the predicted values for the peak value.
#     - the true values to be predicted
#
#     --- Other inputs:
#
#     split_vertical = Boolean. If true, the function calls split_train_and_test() to train
#     only on the first size * len(data_meter) 9-tuples. The rest are used for testing and predictions
#     on those are returned
#
#     size = Float. The fraction of 9-tuples used for training the data
#     """
#
#     if split_vertical == False:
#         # Extract relevant values as arrays
#         y = data_meter["Peak Value"].values
#         X = data_meter.drop(columns=["Peak Value", "Peak Position"]).values
#
#         # Fit Linear Regression
#         model = LinearRegression().fit(X, y)
#         y_pred = model.predict(X)
#
#         return y_pred, y
#
#     else:
#         # Split into training and testing sets
#         data_train, data_test = split_train_and_test(data_meter, size=size)
#
#         # Extract relevant values as arrays
#         y_train = data_train["Peak Value"].values
#         X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values
#
#         y_test = data_test["Peak Value"].values
#         X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values
#
#         # Fit Linear Regression
#         model = LinearRegression().fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         return y_pred, y_test
#
# # Linear Regression globally
# def linregress_global(data_meters: list[pd.DataFrame],
#                      split_horizontal = False,
#                      size = 0.8):
#     """
#     This function applies linear regression to the data for all users (globally) in the following way: data frames
#     are concatenated vertically, so essentially we mix all 9-tuples together. The format is the one returned by
#     construct_feature_matrix_DataFrame() (so after applying the sliding window, with 9-tuples as rows).
#
#     The return will be a tuple with:
#     - an array with the predicted values for the peak value.
#     - the true values to be predicted
#
#     --- Other inputs:
#
#     split_horizontal = Boolean. If true, the function splits the data such that only on the first (size * len(data_meters)) **users**
#     are used for training (but with their whole time series). The rest of the users are used for predictions
#
#     size = Float. The fraction of users used for training the data
#     """
#
#     if split_horizontal == False:
#         # Concatenate all in one DataFrame
#         data = pd.concat(data_meters, ignore_index=True, axis=0)
#
#         # Extract relevant values as arrays
#         y = data["Peak Value"].values
#         X = data.drop(columns=["Peak Value", "Peak Position"]).values
#
#         # Fit Linear Regression
#         model = LinearRegression().fit(X, y)
#         y_pred = model.predict(X)
#
#         return y_pred, y
#
#     else:
#         # Keep the first users only
#         data_train = pd.concat(data_meters[:int(size * len(data_meters))], axis=0, ignore_index=True)
#         data_test = pd.concat(data_meters[int(size * len(data_meters)):], axis=0, ignore_index=True)
#
#         # Extract relevant values as arrays
#         y_train = data_train["Peak Value"].values
#         X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values
#
#         y_test = data_test["Peak Value"].values
#         X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values
#
#         # Fit Linear Regression
#         model = LinearRegression().fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         return y_pred, y_test