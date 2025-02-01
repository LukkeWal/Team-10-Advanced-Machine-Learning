from xgboost import XGBRegressor, XGBClassifier
import pandas as pd
import numpy as np
from PrepareData import get_folds
from sklearn.metrics import root_mean_squared_error, accuracy_score

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


def XgBoost_regression(data_train, data_test, params):

    """

        This function applies XGBoost to the data for all users (globally) in the following way: data frames
        are concatenated vertically, so essentially we mix all 9-tuples together. The format is the one returned by
        construct_feature_matrix_DataFrame() (so after applying the sliding window, with 9-tuples as rows).

        The return will be a tuple with:
         - an array with the predicted values for the peak value.
         - the true values to be predicted

    """

    data_train = pd.concat(data_train, axis=0, ignore_index=True)
    data_test = pd.concat(data_test, axis=0, ignore_index=True)

    y_train = data_train["Peak Value"].values
    X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values

    y_test = data_test["Peak Value"].values
    X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values

    model = XGBRegressor(max_depth=params['max_depth'][0], min_child_weight=params['min_child_weight'][0],
                         learning_rate=params['learning_rate'][0], subsample=params['subsample'][0],
                         colsample_bytree=params['colsample_bytree'][0],
                         reg_lambda=params['reg_lambda'][0], gamma=params['gamma'][0], random_state=42,
                         tree_method='approx')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, y_test

def XgBoost_classifier(data_train, data_test, params):

    """

            This function applies XGBoost to the data for all users (globally) in the following way: data frames
            are concatenated vertically, so essentially we mix all 9-tuples together. The format is the one returned by
            construct_feature_matrix_DataFrame() (so after applying the sliding window, with 9-tuples as rows).

            The return will be a tuple with:
             - an array with the predicted position the peak value.
             - the true position's values to be predicted

        """

    data_train = pd.concat(data_train, axis=0, ignore_index=True)
    data_test = pd.concat(data_test, axis=0, ignore_index=True)

    #convert to categorical data
    data_train["Peak Position"] = data_train["Peak Position"].astype("category")
    data_test["Peak Position"] = data_test["Peak Position"].astype("category")

    y_train = data_train["Peak Position"].values
    X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values

    y_test = data_test["Peak Position"].values
    X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values


    # one-hot encode the target variable
    #encoder = OneHotEncoder(sparse_output=False)
    #y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    #y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    #print(y_train_encoded[0])
    #print(y_test_encoded[0])


    model = XGBClassifier( max_depth = params['max_depth'][0], min_child_weight = params['min_child_weight'][0],
                              learning_rate = params['learning_rate'][0], subsample = params['subsample'][0],colsample_bytree = params['colsample_bytree'][0],
                              reg_lambda = params['reg_lambda'][0],gamma = params['gamma'][0],random_state=42, tree_method = 'approx', objective='multi:softmax')
                              #reg_alpha = params['reg_alpha'][0],reg_lambda = params['reg_lambda'][0],random_state=params['random_state'], enable_categorical=True, # this will use XGBoost's internal voodoo to deal with categorical data
                              #  device="gpu") #remove or change to "cpu" if it's not working

    # default params for testing
    #model = XGBClassifier(enable_categorical=True, device="gpu" )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, y_test

def xgBoost_cv_hyp_search(whichTask,data_train):

    """

        This function performs 3-fold cross validation on 'data' in order to search for the best paramters.
        If whichTask is regression then it will perform CV using rmse
        If whichTask is classification then it will perform CV using accuracy
        They return the best set of paramters that minimizes the cost function and the latter value.

    """

    nSplits = 3
    folds = get_folds(data_train,nSplits)

    maxIter = 100

    best_space = {'max_depth': 0,
                    'min_child_weight': 0,
                    'learning_rate': 0,
                    'subsample': 0,
                    'colsample_bytree': 0,
                    'reg_lambda': 0,
                    'gamma': 0
                    }


    if whichTask == 'regression':
        best_rmse = 10000 # set a very high rmse such that then first paramters' guess will be accepted

        for i in range(maxIter):
            search_space = {'max_depth': np.random.randint(3, 8, 1),
                        'min_child_weight': np.random.uniform(0, 10, 1),
                        'learning_rate': np.random.uniform(0, 0.35, 1),
                        'subsample': np.random.uniform(0.1, 0.6, 1),
                        'colsample_bytree': np.random.uniform(0.1, 1, 1),
                        'reg_lambda': np.random.uniform(0, 3, 1),
                        'gamma': np.random.uniform(0, 3, 1)
                        }

            rmse_results = []

            for k in range(nSplits):
                train_data = folds[0][k]
                test_data = folds[1][k]

                y_pred, y_true = XgBoost_regression(train_data, test_data,search_space )

                rmse_results.append(root_mean_squared_error(y_true, y_pred))

                avg_rmse = np.mean(rmse_results)

            if avg_rmse < best_rmse: # if the average is lower than the best_rmse then we have found a new best set of paramters

                best_rmse = avg_rmse
                best_space = search_space

        return best_space, best_rmse

    if whichTask == 'classification':

        best_accuracy = -1 # set an accuracy smaller than 0 such that then first paramters' guess will be accepted

        for i in range(maxIter):
            search_space = {'max_depth': np.random.randint(3, 8, 1),
                                'min_child_weight': np.random.uniform(0, 10, 1),
                                'learning_rate': np.random.uniform(0, 0.35, 1),
                                'subsample': np.random.uniform(0.1, 0.6, 1),
                                'colsample_bytree': np.random.uniform(0.1, 1, 1),
                                'reg_lambda': np.random.uniform(0, 3, 1),
                                'gamma': np.random.uniform(0, 3, 1)
                                }

            accuracy_results = []

            for k in range(nSplits):
                train_data = folds[0][k]
                test_data = folds[1][k]

                y_pred, y_true = XgBoost_classifier(train_data, test_data, search_space)

                accuracy_results.append(accuracy_score(y_true, y_pred))

                avg_accuracy = np.mean(accuracy_results)

            if avg_accuracy > best_accuracy: # if the average is higher than the best_accuracy then we have found a new best set of paramters
                best_accuracy = avg_accuracy
                best_space = search_space

        return best_space, best_accuracy

