import pandas as pd

from PrepareData import read_data, change_interval_to_one_day, format_data, remove_first_year, split_train_and_test, z_normalize, apply_sliding_window


def load_raw_data() -> pd.DataFrame:
    """
    loads the data from the original data files
    this is slow so the result is stored in "time_series_matrix.csv" for future use
    """
    data = read_data() # read the data from the original files
    data = change_interval_to_one_day(data) # sums all intervals in a day and discards some unneeded columns
    data = format_data(data) # combines all seperate dataframes (1 for each meter) into a single dataframe
    data.to_csv("time_series_matrix.csv") # save results becaues its a lot of data and takes long
    return data

def load_precomputed_data() -> pd.DataFrame:
    """
    Load the data from the precomputed "time_series_matrix.csv" if available
    """
    data = pd.read_csv("time_series_matrix.csv")
    return data

def main():
    load_raw_data()
    return

if __name__ == "__main__":
    main()