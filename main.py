import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error, accuracy_score
from PrepareData import read_data, prev_week_DataFrame,get_folds
from LinearRegression import linregress_meter, linregress_global
from XGBoost import XgBoost_global,XgBoost
from xgboost import XGBRegressor, XGBClassifier
<<<<<<< HEAD
from sklearn.linear_model import LinearRegression
=======
from sklearn.preprocessing import OneHotEncoder
>>>>>>> 204b92c648bfeb41d9183076c734199314bb16df


from pdb import set_trace

def main():
    
    ##### LOAD THE DATA #####

    start_time = time.time()
    data = read_data("ProcessedData")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"processed data in {elapsed_time:.4f} seconds")

    ##### Start with the grid search #####

    # For simplicitly, let's start with a single user, and develop the machinery on it.

    # --- The baseline: linear regression ---

    ## Start with a single user

    # user = 5
    # data_user = data[user]

    # y_pred, y_true = linregress_meter(data_user, split_vertical=True, size=0.8)
    # print(f"RMSE: {root_mean_squared_error(y_pred, y_true)}")

    # # Plot the predictions
    # fig1 = plt.figure(figsize=(6, 6))

    # plt.plot(y_true, color="blue", label="True")
    # plt.plot(y_pred, color="red", label="Prediction")

    # plt.title(f"Meter {user} - local prediction")
    # plt.legend()
    # plt.show()

    # # Plot the residues
    # fig2 = plt.figure(figsize=(6, 6))

    # plt.plot(y_true - y_pred, color="blue", label="True")

    # plt.title(f"Meter {user} - residues")
    # plt.legend()
    # plt.show()

    # fig3 = plt.figure(figsize=(6, 6))

    # plt.hist(y_true - y_pred, color="blue", label="True", bins=100)

    # plt.title(f"Meter {user} - residues")
    # plt.legend()
    # plt.show()

    ## What if we have more users? And compared to prevweek?

    # residue_list = np.array([])
    # residue_list_prev = np.array([])

    # for data_user in data:
    #     y_pred, y_true = linregress_meter(data_user, split_vertical=True, size=0.8)
    #     y_pred_prev, _ = prev_week_DataFrame(data_user)
        
    #     residue_list = np.concatenate((residue_list, y_true - y_pred))
    #     residue_list_prev = np.concatenate((residue_list_prev, y_true - y_pred_prev[-y_true.size:]))
    
    # print(residue_list.mean())
    # print(residue_list.std())

    # print(residue_list_prev.mean())
    # print(residue_list_prev.std())

    # # Intermediate results:

    # # PrevWeek: mean 0.02, std 1.06
    # # LinRegress: mean 0.04, std 0.9

    # fig3 = plt.figure(figsize=(6, 6))

    # plt.hist(residue_list, color="blue", label="LinearRegression", 
    #          bins=100, density=True, alpha=0.8)
    # plt.hist(residue_list_prev, color="red", label="PrevWeek", 
    #          bins=100, density=True, alpha = 0.8)

    # plt.title(f"Pool all users - residues")
    # plt.legend()
    # plt.show()

    ## --- Do global prediction with linear regression

    # y_pred, y_true = linregress_global(data, split_horizontal=True, size=0.8)
    # residues = y_pred - y_true
    
    # print(f"RMSE: {root_mean_squared_error(y_pred, y_true)}")
    # print(residues.mean())
    # print(residues.std())

    # # Intermediate result: mean 0.0008, std 0.9

    # # Plot the predictions
    # fig1 = plt.figure(figsize=(6, 6))

    # plt.plot(y_true, color="blue", label="True")
    # plt.plot(y_pred, color="red", label="Prediction")

    # plt.title(f"Global prediction (test set)")
    # plt.legend()
    # plt.show()

    # # Plot the residues
    # fig2 = plt.figure(figsize=(6, 6))

    # plt.plot(residues, color="blue", label="True")

    # plt.title(f"Global prediction (test set) - residues")
    # plt.legend()
    # plt.show()

    # fig3 = plt.figure(figsize=(6, 6))

    # plt.hist(residues, color="blue", label="True", bins=100)

    # plt.title(f"Global prediction (test set) - residues")
    # plt.legend()
    # plt.show()

    ## --- Do some simple XGBoost regression - global prediction

    # y_pred, y_true = XgBoost_global(data, split_horizontal=True, size=0.8)
    # residues = y_pred - y_true
    
    # print(f"RMSE: {root_mean_squared_error(y_pred, y_true)}")
    # print(residues.mean())
    # print(residues.std())

    # # Intermediate result: mean -0.002, std 0.9

    # # Plot the predictions
    # fig1 = plt.figure(figsize=(6, 6))

    # plt.plot(y_true, color="blue", label="True")
    # plt.plot(y_pred, color="red", label="Prediction")

    # plt.title(f"Global prediction (test set)")
    # plt.legend()
    # plt.show()

    # # Plot the residues
    # fig2 = plt.figure(figsize=(6, 6))

    # plt.plot(residues, color="blue", label="True")

    # plt.title(f"Global prediction (test set) - residues")
    # plt.legend()
    # plt.show()

    # fig3 = plt.figure(figsize=(6, 6))

    # plt.hist(residues, color="blue", label="True", bins=100)

    # plt.title(f"Global prediction (test set) - residues")
    # plt.legend()
    # plt.show()

    def XgBoost_regression(data_train, testData, params):

        data_train = pd.concat(data_train, axis=0, ignore_index=True)
        data_test = pd.concat(testData, axis=0, ignore_index=True)

        y_train = data_train["Peak Value"].values
        X_train = data_train.drop(columns=["Peak Value", "Peak Position"]).values

        y_test = data_test["Peak Value"].values
        X_test = data_test.drop(columns=["Peak Value", "Peak Position"]).values

        model = XGBRegressor( max_depth = params['max_depth'][0], min_child_weight = params['min_child_weight'][0],
                              learning_rate = params['learning_rate'][0], subsample = params['subsample'][0],colsample_bytree = params['colsample_bytree'][0],
                              reg_lambda = params['reg_lambda'][0],gamma = params['gamma'][0],random_state=42, tree_method = 'approx')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred, y_test


    def XgBoost_classifier(data_train, testData, params):

        data_train = pd.concat(data_train, axis=0, ignore_index=True)
        data_test = pd.concat(testData, axis=0, ignore_index=True)

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
<<<<<<< HEAD
                              reg_lambda = params['reg_lambda'][0],gamma = params['gamma'][0],random_state=42, tree_method = 'approx', objective='multi:softmax')
=======
                              reg_alpha = params['reg_alpha'][0],reg_lambda = params['reg_lambda'][0],random_state=params['random_state'], enable_categorical=True, # this will use XGBoost's internal voodoo to deal with categorical data
                                device="gpu") #remove or change to "cpu" if it's not working
>>>>>>> 204b92c648bfeb41d9183076c734199314bb16df

        # default params for testing
        #model = XGBClassifier(enable_categorical=True, device="gpu" )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred, y_test

    def xgBoost_cv_hyp_search(whichTask):



        size = 0.8
        data_train = data[:int(size * len(data))] # for cross validation
        # data_test = data[int(size * len(data)):] # for final testing

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
            best_rmse = 10000

            for i in range(maxIter):
                print(i)
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

                if avg_rmse < best_rmse:

                    best_rmse = avg_rmse
                    best_space = search_space

            return best_space, best_rmse

        if whichTask == 'classification':

            best_accuracy = -1

            for i in range(maxIter):
                print(i)
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

                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_space = search_space

            return best_space, best_accuracy


    def least_squares(data_train, testData):

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


    size = 0.8
    #data_test = data[int(size * len(data)):]  # for final testing
    #data_train = data[:int(size * len(data))]  # for cross validation
    #print(xgBoost_cv_hyp_search('regression'))
    print(xgBoost_cv_hyp_search('classification'))



if __name__ == "__main__":
    main()
