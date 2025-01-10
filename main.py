import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from PrepareData import load_and_save_raw_data, load_precomputed_data, load_and_save_processed_data, read_data
from PrepareData import construct_feature_matrix_array, skip_dates_mask, construct_feature_matrix_DataFrame, prev_week_DataFrame
from DataStatistics import generate_and_visualize_stats
from LinearRegression import linregress_meter, linregress_global
from XGBoost import XgBoost_meter

from pdb import set_trace

def main():
    
    ##### CREATE AND LOAD THE DATA #####
    
    # - Load the data
    
    #start_time = time.time()
    #data = load_and_save_raw_data("LADPUTESTING", "time_series_matrix_TESTING1.csv")
    #generate_and_visualize_stats(data)
    # data = load_precomputed_data("time_series_matrix_TESTING1.csv")
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"loaded data in {elapsed_time:.4f} seconds")
    #generate_and_visualize_stats(data)

    # - Apply sliding window

    # start_time = time.time()
    # data_processed = construct_feature_matrix_DataFrame(data)    # Change to _array for Roberto's function
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"processed data in {elapsed_time:.4f} seconds")

    # - Just load data

    temp = load_and_save_processed_data(precomp_data_dirname="time_series_matrix_TESTING1.csv",
                                        save_data=True)
    temp2 = read_data("ProcessedData")

    print(temp[2].head())
    print(temp2[2].head())
    
    # ##### EXAMPLES OF REGRESSION METHODS #####

    # # - Linear Regression

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

    # # - XGBoost

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

    return

if __name__ == "__main__":
    main()