import time
import numpy as np

from PrepareData import load_and_save_raw_data, load_precomputed_data, construct_feature_matrix_DataFrame, prev_week_DataFrame
from DataStatistics import generate_and_visualize_stats

from pdb import set_trace

def main():
    # - Load the data
    
    start_time = time.time()
    #data = load_and_save_raw_data("LADPU", "time_series_matrix.csv")
    data = load_precomputed_data("time_series_matrix_TESTING.csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"loaded data in {elapsed_time:.4f} seconds")
    #generate_and_visualize_stats(data)

    # - Apply sliding window

    start_time = time.time()
    data_processed = construct_feature_matrix_DataFrame(data)    # Change to _array for Roberto's function
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"processed data in {elapsed_time:.4f} seconds")

    # set_trace()
    # meter_1 = data_processed[0]
    # temp = prev_week_DataFrame(meter_1)
    # print("Stop debugger here")

    return

if __name__ == "__main__":
    main()