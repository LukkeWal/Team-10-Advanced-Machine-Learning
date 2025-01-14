import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from PrepareData import read_data, prev_week_DataFrame
from LinearRegression import linregress_meter, linregress_global
from XGBoost import XgBoost_meter, XgBoost_global

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

    return

if __name__ == "__main__":
    main()