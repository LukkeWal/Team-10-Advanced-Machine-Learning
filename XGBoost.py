from xgboost import XGBRegressor
import numpy as np
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

def XgBoost_meter(X_train, X_test, y_train, y_test):

    model = XGBRegressor() # specify paramters
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    RMSE = root_mean_squared_error(y_test, y_pred)
    scores = [RMSE]

    return y_pred, scores