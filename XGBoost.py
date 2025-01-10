from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import root_mean_squared_error

def XgBoost_meter(X_train, X_test, y_train, y_test):

    model = XGBRegressor() # specify paramters
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    RMSE = root_mean_squared_error(y_test, y_pred)
    scores = [RMSE]

    return y_pred, scores