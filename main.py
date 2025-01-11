import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from PrepareData import read_data
from LinearRegression import linregress_meter, linregress_global
from XGBoost import XgBoost_meter

from pdb import set_trace

def main():
    
    ##### LOAD THE DATA #####

    start_time = time.time()
    data = read_data("ProcessedData")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"processed data in {elapsed_time:.4f} seconds")

    ##### the rest... #####

    return

if __name__ == "__main__":
    main()