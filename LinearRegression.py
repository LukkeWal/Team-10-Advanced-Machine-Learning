import pandas as pd
from sklearn.linear_model import LinearRegression

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